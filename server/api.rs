use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MODEL: &str = "qwasar-qwen38-27b";
static COUNTER: AtomicU64 = AtomicU64::new(0);

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

pub fn normalize_messages(messages: &[Value]) -> Result<Vec<Value>, String> {
    let mut normalized = Vec::new();
    for message in messages {
        let role = message["role"].as_str().ok_or("message requires role")?;
        let role = if role == "developer" { "system" } else { role };
        if !matches!(role, "system" | "user" | "assistant" | "tool") {
            return Err("unsupported message role".into());
        }
        let content = text_content(&message["content"])?;
        if role == "system" && !normalized.is_empty() {
            return Err("system/developer message must be first".into());
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

fn responses_input(input: &Value) -> Result<Vec<Value>, String> {
    if let Some(text) = input.as_str() {
        return Ok(vec![json!({"role":"user","content":text})]);
    }
    let mut messages = Vec::new();
    for item in input.as_array().ok_or("input must be text or an array")? {
        match item["type"].as_str().unwrap_or("message") {
            "message" => messages.push(item.clone()),
            "function_call_output" => messages.push(json!({"role":"tool","tool_call_id":item["call_id"].as_str().ok_or("call_id required")?,"content":text_content(&item["output"])?})),
            "function_call" => messages.push(json!({"role":"assistant","content":"","tool_calls":[{"id":item["call_id"],"type":"function","function":{"name":item["name"],"arguments":item["arguments"]}}]})),
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
        "store",
        "truncation",
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
    if body.get("store").is_some_and(|value| value != &json!(true)) {
        return Err("v1 requires durable store=true".into());
    }
    if body
        .get("truncation")
        .is_some_and(|value| value != "disabled")
    {
        return Err("automatic truncation is not supported".into());
    }
    if let Some(options) = body.get("stream_options") {
        if !options.is_object()
            || options
                .as_object()
                .unwrap()
                .keys()
                .any(|key| key != "include_usage")
            || options
                .get("include_usage")
                .is_some_and(|value| !value.is_boolean())
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
        if options.keys().any(|key| key != "effort")
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
    let messages = normalize_messages(&messages)?;
    if !messages.iter().any(|message| message["role"] == "user") {
        return Err("at least one user message required".into());
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
    for tool in tool_list {
        if responses && tool["type"] == "function" && tool.get("function").is_none() {
            let mut function = tool.clone();
            function.as_object_mut().unwrap().remove("type");
            *tool = json!({"type":"function","function":function});
        }
        if tool["type"] != "function"
            || !tool["function"]["name"].is_string()
            || !tool["function"]["parameters"].is_object()
        {
            return Err("only function tools with schemas are supported".into());
        }
    }
    let mut choice = body.get("tool_choice").cloned().unwrap_or(json!("auto"));
    if responses && choice["type"] == "function" && choice.get("function").is_none() {
        choice = json!({"type":"function","function":{"name":choice["name"]}});
    }
    if !matches!(choice.as_str(), Some("auto" | "none" | "required"))
        && !(choice["type"] == "function" && choice["function"]["name"].is_string())
    {
        return Err("unsupported tool_choice".into());
    }
    Ok(
        json!({"messages":messages,"tools":tools,"thinking":thinking,"temperature":temperature,"top_p":top_p,"seed":seed,"max_tokens":max_tokens,"tool_choice":choice}),
    )
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
