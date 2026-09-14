use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MODEL: &str = "qwasar-qwen38-27b";
const CODEX_INSTRUCTIONS: &str = "You are Codex, a capable AI software engineering assistant. You help users with coding tasks including reading, writing, debugging, refactoring, and explaining code. You can use tools to read and write files, run shell commands, and search the codebase. Be concise, precise, and proactive. When you make changes, briefly explain what you did and why. Prefer minimal, correct changes that follow existing project conventions.";
static COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn models_payload() -> Value {
    json!({
        "object":"list",
        "data":[{"id":MODEL,"object":"model","owned_by":"qwasar","context_window":262144}],
        "models":[{
            "slug":MODEL,
            "display_name":"Qwasar Qwen3.8-27B",
            "description":"Qwasar hybrid (RTX 5090, NVIDIA64+PRIMS+ATT64), 262k ctx, native image input.",
            "base_instructions":CODEX_INSTRUCTIONS,
            "supported_reasoning_levels":[
                {"effort":"low","description":"Low reasoning"},
                {"effort":"medium","description":"Medium reasoning"},
                {"effort":"high","description":"High reasoning"},
                {"effort":"xhigh","description":"Extra-high reasoning"}
            ],
            "shell_type":"unified_exec",
            "visibility":"list",
            "supported_in_api":true,
            "priority":0,
            "support_verbosity":false,
            "truncation_policy":{"mode":"bytes","limit":10000},
            "experimental_supported_tools":[],
            "context_window":262144,
            "max_context_window":262144,
            "effective_context_window_percent":95,
            "input_modalities":["text","image"]
        }]
    })
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

pub fn new_id(prefix: &str) -> String {
    format!(
        "{prefix}_{:x}_{:x}_{:x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

pub const IMAGE_MEDIA_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/webp", "image/gif"];
pub const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_IMAGES: usize = 16;

/// Image blobs referenced by a request, keyed by lowercase SHA-256 hex. `data`
/// is `None` when the request only referenced a hash (Responses parents) and
/// the bytes must come from the store.
pub type Images = std::collections::BTreeMap<String, (String, Option<Vec<u8>>)>;

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn image_part(url: &str, images: &mut Images) -> Result<Value, String> {
    use base64::Engine;
    let payload = url
        .strip_prefix("data:")
        .ok_or("images must be inline data URLs; remote URLs are not fetched")?;
    let (header, data) = payload
        .split_once(',')
        .ok_or("malformed image data URL")?;
    let (media_type, encoding) = header.split_once(';').ok_or("image data URL must be base64")?;
    if encoding != "base64" {
        return Err("image data URL must be base64".into());
    }
    if !IMAGE_MEDIA_TYPES.contains(&media_type) {
        return Err(format!("unsupported image media type: {media_type}"));
    }
    if data.len() > MAX_IMAGE_BYTES / 3 * 4 + 4 {
        return Err(format!("image exceeds {MAX_IMAGE_BYTES} bytes"));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .map_err(|_| "image data is not valid base64")?;
    if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
        return Err(format!("image must contain 1..{MAX_IMAGE_BYTES} bytes"));
    }
    let sha256 = sha256_hex(&bytes);
    images.insert(sha256.clone(), (media_type.to_string(), Some(bytes)));
    Ok(json!({"type":"image","sha256":sha256,"media_type":media_type}))
}

fn canonical_image_part(part: &Value, images: &mut Images) -> Result<Value, String> {
    let sha256 = part["sha256"]
        .as_str()
        .filter(|hex| hex.len() == 64 && hex.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f')))
        .ok_or("image part requires a lowercase sha256 hex")?;
    let media_type = part["media_type"]
        .as_str()
        .filter(|kind| IMAGE_MEDIA_TYPES.contains(kind))
        .ok_or("image part requires a supported media_type")?;
    images
        .entry(sha256.to_string())
        .or_insert((media_type.to_string(), None));
    Ok(json!({"type":"image","sha256":sha256,"media_type":media_type}))
}

fn image_url(part: &Value) -> Result<&str, String> {
    match &part["image_url"] {
        Value::String(url) => Ok(url),
        Value::Object(object) => object["url"].as_str().ok_or("image_url requires url".into()),
        _ => Err("image part requires image_url".into()),
    }
}

/// Normalizes user content into either a plain string (text only) or a parts
/// array mixing `text` and canonical `image` parts. Only user messages may
/// carry images; every other role goes through `text_content`.
pub fn user_content(value: &Value, images: &mut Images) -> Result<Value, String> {
    let Value::Array(parts) = value else {
        return Ok(json!(text_content(value)?));
    };
    let mut output: Vec<Value> = Vec::new();
    let mut has_image = false;
    for part in parts {
        let normalized = match part["type"].as_str() {
            Some("text" | "input_text" | "output_text") => {
                json!({"type":"text","text":part["text"].as_str().ok_or("text part requires text")?})
            }
            Some("image_url" | "input_image") => {
                has_image = true;
                image_part(image_url(part)?, images)?
            }
            Some("image") if part.get("sha256").is_some() => {
                has_image = true;
                canonical_image_part(part, images)?
            }
            _ => return Err("only text and inline image parts are supported".into()),
        };
        match (output.last_mut(), &normalized) {
            (Some(last), Value::Object(_)) if last["type"] == "text" && normalized["type"] == "text" => {
                let joined = format!("{}{}", last["text"].as_str().unwrap(), normalized["text"].as_str().unwrap());
                last["text"] = json!(joined);
            }
            _ => output.push(normalized),
        }
    }
    if !has_image {
        return Ok(json!(output.iter().map(|part| part["text"].as_str().unwrap_or("")).collect::<String>()));
    }
    Ok(Value::Array(output))
}

/// Replaces inline image data URLs anywhere in a request body with canonical
/// hash references so stored requests stay small and idempotency comparisons
/// remain deterministic. Invalid parts are left untouched; `prepare` rejects them.
pub fn redact_images(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(redact_images).collect()),
        Value::Object(object) => {
            if matches!(object.get("type").and_then(Value::as_str), Some("image_url" | "input_image")) {
                let mut scratch = Images::new();
                if let Ok(part) = image_url(value).and_then(|url| image_part(url, &mut scratch)) {
                    return part;
                }
            }
            Value::Object(object.iter().map(|(key, item)| (key.clone(), redact_images(item))).collect())
        }
        other => other.clone(),
    }
}

pub fn text_content(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if !matches!(
                    part["type"].as_str(),
                    Some("text" | "input_text" | "output_text")
                ) {
                    return Err("only text content is supported".into());
                }
                text.push_str(part["text"].as_str().ok_or("text part requires text")?);
            }
            Ok(text)
        }
        _ => Err("content must be text or text parts".into()),
    }
}

pub fn normalize_messages(messages: &[Value], images: &mut Images) -> Result<Vec<Value>, String> {
    let mut normalized: Vec<Value> = Vec::new();
    let mut system_index: Option<usize> = None;
    for message in messages {
        let role = message["role"].as_str().ok_or("message requires role")?;
        let role = if role == "developer" { "system" } else { role };
        if !matches!(role, "system" | "user" | "assistant" | "tool") {
            return Err("unsupported message role".into());
        }
        if role == "user" {
            let content = user_content(&message["content"], images)?;
            normalized.push(json!({"role":"user","content":content}));
            continue;
        }
        let content = text_content(&message["content"])
            .map_err(|error| if error.starts_with("only text") { "images are only supported in user messages".to_string() } else { error })?;
        if role == "system" {
            if let Some(index) = system_index {
                if !content.is_empty() {
                    let existing = normalized[index]["content"].as_str().unwrap_or("");
                    normalized[index]["content"] = json!(if existing.is_empty() {
                        content
                    } else {
                        format!("{existing}\n\n{content}")
                    });
                }
            } else {
                normalized.insert(0, json!({"role":"system","content":content}));
                system_index = Some(0);
            }
            continue;
        }
        let mut entry = json!({"role":role,"content":content});
        if role == "assistant" {
            if let Some(reasoning) = message.get("reasoning_content") {
                entry["reasoning_content"] = json!(text_content(reasoning)?);
            }
            if let Some(calls) = message.get("tool_calls") {
                if !calls.is_array() {
                    return Err("tool_calls must be an array".into());
                }
                entry["tool_calls"] = calls.clone();
            }
        }
        if role == "tool" {
            entry["tool_call_id"] = json!(
                message["tool_call_id"]
                    .as_str()
                    .ok_or("tool result requires tool_call_id")?
            );
        }
        normalized.push(entry);
    }
    Ok(normalized)
}

pub fn history_key(messages: &[Value]) -> String {
    let normalized: Vec<Value> = messages
        .iter()
        .filter(|message| message["role"] != "system" && message["role"] != "developer")
        .map(|message| {
            let mut message = message.clone();
            if let Some(object) = message.as_object_mut() {
                object.remove("reasoning_content");
                if object.get("content").is_none_or(Value::is_null) {
                    object.insert("content".into(), json!(""));
                }
                if object
                    .get("tool_calls")
                    .is_some_and(|calls| calls.as_array().is_some_and(Vec::is_empty))
                {
                    object.remove("tool_calls");
                }
            }
            if let Some(calls) = message["tool_calls"].as_array_mut() {
                for call in calls {
                    if let Some(arguments) = call["function"]["arguments"].as_str() {
                        if let Ok(parsed) = serde_json::from_str::<Value>(arguments) {
                            call["function"]["arguments"] = parsed;
                        }
                    }
                }
            }
            message
        })
        .collect();
    Sha256::digest(serde_json::to_vec(&normalized).unwrap())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn tool_call_id(item: &Value) -> Result<String, String> {
    item["call_id"]
        .as_str()
        .or_else(|| item["id"].as_str())
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .ok_or_else(|| "call_id required".into())
}

fn json_argument_string(value: &Value) -> String {
    match value {
        Value::String(text) if text.trim().is_empty() => "{}".into(),
        Value::String(text) if serde_json::from_str::<Value>(text).is_ok() => text.clone(),
        Value::String(text) => json!({"input": text}).to_string(),
        Value::Null => "{}".into(),
        other => other.to_string(),
    }
}

fn assistant_function_call(id: &str, name: &str, arguments: String) -> Value {
    json!({"role":"assistant","content":"","tool_calls":[{"id":id,"type":"function","function":{"name":name,"arguments":arguments}}]})
}

fn function_tool(name: &str, description: Option<&str>, parameters: Value) -> Value {
    json!({"type":"function","function":{"name":name,"description":description.unwrap_or(""),"parameters":parameters}})
}

fn object_parameters(properties: Value, required: &[&str]) -> Value {
    json!({"type":"object","properties":properties,"required":required})
}

fn normalize_tool(tool: &Value, responses: bool) -> Result<Value, String> {
    match tool["type"].as_str().unwrap_or("function") {
        "function" => {
            let function = if tool.get("function").is_some() {
                tool["function"].clone()
            } else if responses {
                let mut function = tool.clone();
                if let Some(object) = function.as_object_mut() {
                    object.remove("type");
                }
                function
            } else {
                return Err("only function tools with schemas are supported".into());
            };
            let name = function["name"]
                .as_str()
                .ok_or("only function tools with schemas are supported")?;
            let mut parameters = function
                .get("parameters")
                .cloned()
                .unwrap_or(json!({"type":"object","properties":{}}));
            if !parameters.is_object() {
                return Err("only function tools with schemas are supported".into());
            }
            if parameters.get("type").is_none() {
                parameters["type"] = json!("object");
            }
            Ok(function_tool(
                name,
                function["description"].as_str(),
                parameters,
            ))
        }
        "custom" => Ok(function_tool(
            tool["name"]
                .as_str()
                .ok_or("only function tools with schemas are supported")?,
            tool["description"].as_str(),
            object_parameters(
                json!({"input":{"type":"string","description":"Freeform tool input"}}),
                &["input"],
            ),
        )),
        "local_shell" => Ok(function_tool(
            "local_shell",
            Some("Run a local shell command."),
            object_parameters(
                json!({
                    "command":{"type":"array","items":{"type":"string"}},
                    "working_directory":{"type":"string"},
                    "timeout_ms":{"type":"number"}
                }),
                &["command"],
            ),
        )),
        "web_search" => Ok(function_tool(
            "web_search",
            Some("Search the web."),
            object_parameters(json!({"query":{"type":"string"}}), &["query"]),
        )),
        "image_generation" => Ok(function_tool(
            "image_generation",
            Some("Generate an image."),
            object_parameters(json!({"prompt":{"type":"string"}}), &["prompt"]),
        )),
        "tool_search" => Ok(function_tool(
            "tool_search",
            tool["description"].as_str(),
            tool.get("parameters")
                .cloned()
                .unwrap_or(json!({"type":"object","properties":{}})),
        )),
        _ => {
            let name = tool["name"]
                .as_str()
                .or_else(|| tool["function"]["name"].as_str())
                .ok_or("only function tools with schemas are supported")?;
            let parameters = tool
                .get("parameters")
                .cloned()
                .or_else(|| tool.pointer("/function/parameters").cloned())
                .unwrap_or(json!({"type":"object","properties":{}}));
            if !parameters.is_object() {
                return Err("only function tools with schemas are supported".into());
            }
            Ok(function_tool(name, tool["description"].as_str(), parameters))
        }
    }
}

fn responses_input(input: &Value) -> Result<Vec<Value>, String> {
    if let Some(text) = input.as_str() {
        return Ok(vec![json!({"role":"user","content":text})]);
    }
    let mut messages = Vec::new();
    for item in input.as_array().ok_or("input must be text or an array")? {
        match item["type"].as_str().unwrap_or("message") {
            "message" => messages.push(item.clone()),
            "function_call_output" | "custom_tool_call_output" => messages.push(json!({
                "role":"tool",
                "tool_call_id":tool_call_id(item)?,
                "content":text_content(item.get("output").unwrap_or(&Value::Null))?
            })),
            "function_call" => messages.push(assistant_function_call(
                &tool_call_id(item)?,
                item["name"].as_str().ok_or("function call requires name")?,
                json_argument_string(&item["arguments"]),
            )),
            "custom_tool_call" => messages.push(assistant_function_call(
                &tool_call_id(item)?,
                item["name"].as_str().ok_or("custom tool call requires name")?,
                json_argument_string(&item["input"]),
            )),
            "local_shell_call" => messages.push(assistant_function_call(
                &tool_call_id(item)?,
                "local_shell",
                json_argument_string(&item["action"]),
            )),
            "reasoning" | "web_search_call" | "image_generation_call" => {}
            _ => return Err("unsupported Responses input item".into()),
        }
    }
    Ok(messages)
}

pub fn prepare(body: &Value, responses: bool, parent: Option<&Value>) -> Result<Value, String> {
    let object = body.as_object().ok_or("request body must be an object")?;
    let allowed = [
        "model",
        "messages",
        "input",
        "instructions",
        "previous_response_id",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "tools",
        "tool_choice",
        "chat_template_kwargs",
        "reasoning_effort",
        "reasoning",
        "reasoning_budget_tokens",
        "store",
        "truncation",
        "parallel_tool_calls",
        "metadata",
        "include",
        "prompt_cache_key",
        "text",
        "client_metadata",
        "service_tier",
        "access_programs",
        "generate",
    ];
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unsupported parameter: {key}"));
        }
    }
    if body["model"].as_str() != Some(MODEL) {
        return Err(format!("model must be {MODEL}"));
    }
    if body.get("stream").is_some_and(|value| !value.is_boolean()) {
        return Err("stream must be boolean".into());
    }
    if body.get("store").is_some_and(|value| !value.is_boolean()) {
        return Err("store must be boolean".into());
    }
    if body
        .get("truncation")
        .is_some_and(|value| value != "disabled")
    {
        return Err("automatic truncation is not supported".into());
    }
    if let Some(options) = body.get("stream_options") {
        let options = options.as_object().ok_or("unsupported stream_options")?;
        if options
            .keys()
            .any(|key| key != "include_usage" && key != "reasoning_summary_delivery")
            || options
                .get("include_usage")
                .is_some_and(|value| !value.is_boolean())
            || options
                .get("reasoning_summary_delivery")
                .is_some_and(|value| !value.is_string())
        {
            return Err("unsupported stream_options".into());
        }
    }
    if let Some(effort) = body.get("reasoning_effort") {
        if !effort.is_string() || body.get("reasoning").is_some() {
            return Err("reasoning_effort must be a string and cannot accompany reasoning".into());
        }
    }
    if let Some(reasoning) = body.get("reasoning") {
        let options = reasoning.as_object().ok_or("reasoning must be an object")?;
        if options
            .keys()
            .any(|key| key != "effort" && key != "summary" && key != "context")
            || options
                .get("effort")
                .is_some_and(|value| !value.is_string())
        {
            return Err("only a string reasoning.effort is supported".into());
        }
    }
    let mut messages = Vec::new();
    if responses {
        if body.get("messages").is_some() {
            return Err("Responses uses input, not messages".into());
        }
        if let Some(instructions) = body.get("instructions") {
            messages.push(json!({"role":"system","content":instructions.as_str().ok_or("instructions must be text")?}));
        }
        if let Some(parent) = parent {
            messages.extend(
                parent["messages"]
                    .as_array()
                    .ok_or("invalid parent snapshot")?
                    .iter()
                    .filter(|message| message["role"] != "system")
                    .cloned(),
            );
        }
        messages.extend(responses_input(body.get("input").ok_or("input required")?)?);
    } else {
        if body.get("previous_response_id").is_some()
            || body.get("input").is_some()
            || body.get("instructions").is_some()
        {
            return Err("Chat Completions uses full messages".into());
        }
        messages = body["messages"]
            .as_array()
            .ok_or("messages must be an array")?
            .clone();
    }
    let mut images = Images::new();
    let messages = normalize_messages(&messages, &mut images)?;
    if !messages.iter().any(|message| message["role"] == "user") {
        return Err("at least one user message required".into());
    }
    let image_parts = messages
        .iter()
        .filter_map(|message| message["content"].as_array())
        .flatten()
        .filter(|part| part["type"] == "image")
        .count();
    if image_parts > MAX_IMAGES {
        return Err(format!("at most {MAX_IMAGES} images per request"));
    }
    let mut pending = std::collections::HashSet::new();
    let mut seen = std::collections::HashSet::new();
    for message in &messages {
        if let Some(calls) = message["tool_calls"].as_array() {
            for call in calls {
                let call_id = call["id"]
                    .as_str()
                    .filter(|id| !id.is_empty())
                    .ok_or("tool call requires id")?;
                if !seen.insert(call_id) {
                    return Err("duplicate tool call id".into());
                }
                if call["type"] != "function"
                    || !call["function"]["name"].is_string()
                    || !call["function"]["arguments"].is_string()
                {
                    return Err("malformed tool call".into());
                }
                let arguments: Value =
                    serde_json::from_str(call["function"]["arguments"].as_str().unwrap())
                        .map_err(|_| "tool arguments must be JSON")?;
                if !arguments.is_object() {
                    return Err("tool arguments must be object".into());
                }
                pending.insert(call_id);
            }
        }
        if message["role"] == "tool" && !pending.remove(message["tool_call_id"].as_str().unwrap()) {
            return Err("tool result does not match a pending call".into());
        }
    }
    if !pending.is_empty() {
        return Err("all pending tool calls need results before generation".into());
    }
    let mut thinking = body
        .get("reasoning_effort")
        .or_else(|| {
            body.get("reasoning")
                .and_then(|reasoning| reasoning.get("effort"))
        })
        .and_then(Value::as_str)
        .unwrap_or("medium")
        .to_string();
    if let Some(kwargs) = body.get("chat_template_kwargs") {
        let kwargs = kwargs
            .as_object()
            .ok_or("chat_template_kwargs must be object")?;
        if kwargs
            .keys()
            .any(|key| key != "enable_thinking" && key != "preserve_thinking")
        {
            return Err("unsupported chat template option".into());
        }
        if kwargs
            .get("preserve_thinking")
            .is_some_and(|value| value != &json!(true))
        {
            return Err("exact sessions require preserve_thinking=true".into());
        }
        if let Some(enabled) = kwargs.get("enable_thinking") {
            if !enabled.is_boolean() {
                return Err("enable_thinking must be boolean".into());
            }
            if enabled == &json!(false) {
                thinking = "off".into();
            }
        }
    }
    thinking = match thinking.as_str() {
        "minimal" => "low",
        "high" | "max" => "xhigh",
        other => other,
    }
    .to_string();
    if !["off", "low", "medium", "xhigh"].contains(&thinking.as_str()) {
        return Err("unsupported reasoning effort".into());
    }
    let limits: Vec<&Value> = ["max_tokens", "max_completion_tokens", "max_output_tokens"]
        .iter()
        .filter_map(|key| body.get(key))
        .collect();
    if limits.len() > 1 {
        return Err("specify only one output token limit".into());
    }
    let max_tokens = match limits.first() {
        Some(value) => value
            .as_u64()
            .ok_or("output limit must be a positive integer")?,
        None => 4096,
    };
    if max_tokens == 0 || max_tokens > 32768 {
        return Err("output limit must be 1..32768".into());
    }
    let temperature = body
        .get("temperature")
        .map_or(Ok(if thinking == "off" { 0.7 } else { 1.0 }), |value| {
            value.as_f64().ok_or("temperature must be numeric")
        })?;
    let top_p = body
        .get("top_p")
        .map_or(Ok(if thinking == "off" { 0.8 } else { 0.95 }), |value| {
            value.as_f64().ok_or("top_p must be numeric")
        })?;
    if !(0.0..=2.0).contains(&temperature) || top_p <= 0.0 || top_p > 1.0 {
        return Err("invalid sampling range".into());
    }
    let seed = body.get("seed").map_or(Ok(42), |value| {
        value.as_u64().ok_or("seed must be nonnegative integer")
    })?;
    let mut tools = body.get("tools").cloned().unwrap_or(json!([]));
    let tool_list = tools.as_array_mut().ok_or("tools must be array")?;
    if tool_list.len() > 128 {
        return Err("at most 128 tools supported".into());
    }
    for tool in tool_list.iter_mut() {
        *tool = normalize_tool(tool, responses)?;
    }
    let mut choice = body.get("tool_choice").cloned().unwrap_or(json!("auto"));
    if responses && choice["type"] == "function" && choice.get("function").is_none() {
        choice = json!({"type":"function","function":{"name":choice["name"]}});
    }
    if responses && choice["type"] == "custom" && choice["name"].is_string() {
        choice = json!({"type":"function","function":{"name":choice["name"]}});
    }
    if !matches!(choice.as_str(), Some("auto" | "none" | "required"))
        && !(choice["type"] == "function" && choice["function"]["name"].is_string())
    {
        return Err("unsupported tool_choice".into());
    }
    let mut prepared = json!({"messages":messages,"tools":tools,"thinking":thinking,"temperature":temperature,"top_p":top_p,"seed":seed,"max_tokens":max_tokens,"tool_choice":choice});
    if !images.is_empty() {
        use base64::Engine;
        prepared["images"] = Value::Object(
            images
                .into_iter()
                .map(|(sha256, (media_type, bytes))| {
                    let data = bytes.map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes));
                    (sha256, json!({"media_type":media_type,"data":data}))
                })
                .collect(),
        );
    }
    if let Some(value) = object.get("reasoning_budget_tokens") {
        let budget = value
            .as_u64()
            .ok_or("reasoning_budget_tokens must be a nonnegative integer")?;
        if budget > 32768 {
            return Err("reasoning_budget_tokens must be 0..32768".into());
        }
        prepared["reasoning_budget_tokens"] = json!(budget);
    }
    Ok(prepared)
}

pub fn finish_reason(terminal: &Value) -> &'static str {
    if terminal["status"] == "incomplete" {
        "length"
    } else if terminal["message"]["tool_calls"]
        .as_array()
        .is_some_and(|calls| !calls.is_empty())
    {
        "tool_calls"
    } else {
        "stop"
    }
}

pub fn chat_result(id: &str, terminal: &Value) -> Value {
    json!({"id":id,"object":"chat.completion","created":terminal["created_at"].as_u64().unwrap_or_else(now),"model":MODEL,
        "choices":[{"index":0,"message":terminal["message"],"finish_reason":finish_reason(terminal)}],
        "usage":terminal["usage"],"qwasar_metrics":terminal["metrics"],"status":terminal["status"]})
}

pub fn response_result(id: &str, terminal: &Value) -> Value {
    let mut output = Vec::new();
    if let Some(reasoning) = terminal["message"]["reasoning_content"]
        .as_str()
        .filter(|text| !text.is_empty())
    {
        output.push(json!({"id":format!("rs_{id}"),"type":"reasoning","summary":[{"type":"summary_text","text":reasoning}]}));
    }
    if let Some(content) = terminal["message"]["content"]
        .as_str()
        .filter(|text| !text.is_empty())
    {
        output.push(json!({"id":format!("msg_{id}"),"type":"message","role":"assistant","status":terminal["status"],"content":[{"type":"output_text","text":content,"annotations":[]}]}));
    }
    if let Some(calls) = terminal["message"]["tool_calls"].as_array() {
        for call in calls {
            output.push(json!({"id":format!("fc_{}",call["id"].as_str().unwrap_or("unknown")),"type":"function_call","status":"completed","call_id":call["id"],"name":call["function"]["name"],"arguments":call["function"]["arguments"]}));
        }
    }
    json!({"id":id,"object":"response","created_at":terminal["created_at"].as_u64().unwrap_or_else(now),"model":MODEL,"status":terminal["status"],"output":output,
        "error":terminal["error"],"incomplete_details":if terminal["status"] == "incomplete" {json!({"reason":"max_output_tokens"})} else {Value::Null},
        "usage":{"input_tokens":terminal["usage"]["prompt_tokens"],"output_tokens":terminal["usage"]["completion_tokens"],"total_tokens":terminal["usage"]["total_tokens"],"input_tokens_details":terminal["usage"]["prompt_tokens_details"]},"qwasar_metrics":terminal["metrics"]})
}

pub fn chat_chunk(id: &str, delta: Value, finish: Value) -> Value {
    json!({"id":id,"object":"chat.completion.chunk","created":now(),"model":MODEL,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
}
