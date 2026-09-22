use crate::api::{self, MODEL};
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const ALLOWED: [&str; 19] = [
    "model",
    "messages",
    "system",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stream",
    "stop_sequences",
    "tools",
    "tool_choice",
    "thinking",
    "metadata",
    "service_tier",
    "cache_control",
    "container",
    "mcp_servers",
    "context_management",
    "output_config",
];

pub fn accepted_model(model: &str) -> bool {
    model == MODEL || model == "qwarz" || model.starts_with("claude-")
}

pub fn http_error(status: u16, code: &str, message: &str) -> Response {
    let kind = match code {
        "invalid_request"
        | "invalid_parent"
        | "ambiguous_history"
        | "unknown_image"
        | "invalid_idempotency_key" => "invalid_request_error",
        "worker_unavailable" | "busy" => "overloaded_error",
        _ => "api_error",
    };
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(json!({"type":"error","error":{"type":kind,"message":message}})),
    )
        .into_response()
}

pub fn prepare(body: &Value) -> Result<Value, String> {
    api::prepare(&to_chat(body)?, false, None)
}

pub fn count_tokens(body: &Value) -> Result<u64, String> {
    Ok(estimate_tokens(&prepare(body)?))
}

pub fn to_chat(body: &Value) -> Result<Value, String> {
    let object = body.as_object().ok_or("request body must be an object")?;
    for key in object.keys() {
        if !ALLOWED.contains(&key.as_str()) {
            return Err(format!("unsupported parameter: {key}"));
        }
    }
    let model = body["model"].as_str().ok_or("model required")?;
    if !accepted_model(model) {
        return Err(format!("model must be {MODEL}"));
    }
    if body.get("stream").is_some_and(|value| !value.is_boolean()) {
        return Err("stream must be boolean".into());
    }
    let mut chat = json!({
        "model": MODEL,
        "messages": anthropic_messages(body)?,
        "tools": anthropic_tools(body.get("tools"))?,
        "tool_choice": anthropic_tool_choice(body.get("tool_choice"))?,
    });
    if let Some(value) = body.get("max_tokens") {
        chat["max_tokens"] = value.clone();
    }
    if let Some(value) = body.get("temperature") {
        chat["temperature"] = value.clone();
    }
    if let Some(value) = body.get("top_p") {
        chat["top_p"] = value.clone();
    }
    if let Some(value) = body.get("stream") {
        chat["stream"] = value.clone();
    }
    apply_thinking(&mut chat, body.get("thinking"))?;
    Ok(chat)
}

pub fn result(id: &str, model: &str, terminal: &Value) -> Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": output_blocks(terminal),
        "stop_reason": stop_reason(terminal),
        "stop_sequence": Value::Null,
        "usage": usage(terminal)
    })
}

pub fn stop_reason(terminal: &Value) -> &'static str {
    match api::finish_reason(terminal) {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

fn usage(terminal: &Value) -> Value {
    json!({
        "input_tokens": terminal["usage"]["prompt_tokens"],
        "output_tokens": terminal["usage"]["completion_tokens"],
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": terminal["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0)
    })
}

fn output_blocks(terminal: &Value) -> Vec<Value> {
    let mut blocks = Vec::new();
    if let Some(thinking) = terminal["message"]["reasoning_content"]
        .as_str()
        .filter(|text| !text.is_empty())
    {
        blocks.push(json!({
            "type": "thinking",
            "thinking": thinking,
            "signature": signature(thinking)
        }));
    }
    if let Some(text) = terminal["message"]["content"]
        .as_str()
        .filter(|text| !text.is_empty())
    {
        blocks.push(json!({"type":"text","text":text}));
    }
    if let Some(calls) = terminal["message"]["tool_calls"].as_array() {
        for call in calls {
            let arguments = call["function"]["arguments"].as_str().unwrap_or("{}");
            let input = serde_json::from_str::<Value>(arguments).unwrap_or(json!({}));
            blocks.push(json!({
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": input
            }));
        }
    }
    blocks
}

fn signature(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn apply_thinking(chat: &mut Value, thinking: Option<&Value>) -> Result<(), String> {
    let Some(thinking) = thinking else {
        return Ok(());
    };
    let options = thinking.as_object().ok_or("thinking must be an object")?;
    match options.get("type").and_then(Value::as_str) {
        Some("disabled") => {
            chat["reasoning_effort"] = json!("off");
        }
        Some("enabled" | "adaptive") => {
            chat["reasoning_effort"] = json!("xhigh");
            if let Some(budget) = options.get("budget_tokens") {
                chat["reasoning_budget_tokens"] = budget.clone();
            }
        }
        _ => return Err("unsupported thinking type".into()),
    }
    Ok(())
}

fn anthropic_tools(tools: Option<&Value>) -> Result<Value, String> {
    let Some(tools) = tools else {
        return Ok(json!([]));
    };
    let tools = tools.as_array().ok_or("tools must be an array")?;
    let mut output = Vec::new();
    for tool in tools {
        let name = tool["name"]
            .as_str()
            .ok_or("tool requires name")?;
        let parameters = tool
            .get("input_schema")
            .cloned()
            .unwrap_or(json!({"type":"object","properties":{}}));
        if !parameters.is_object() {
            return Err("input_schema must be an object".into());
        }
        output.push(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": tool["description"].as_str().unwrap_or(""),
                "parameters": parameters
            }
        }));
    }
    Ok(Value::Array(output))
}

fn anthropic_tool_choice(choice: Option<&Value>) -> Result<Value, String> {
    let Some(choice) = choice else {
        return Ok(json!("auto"));
    };
    match choice["type"].as_str() {
        Some("auto") => Ok(json!("auto")),
        Some("any") => Ok(json!("required")),
        Some("none") => Ok(json!("none")),
        Some("tool") => {
            let name = choice["name"].as_str().ok_or("tool_choice.tool requires name")?;
            Ok(json!({"type":"function","function":{"name":name}}))
        }
        _ => Err("unsupported tool_choice".into()),
    }
}

fn anthropic_text(value: &Value) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if let Some(piece) = part["text"].as_str() {
                    text.push_str(piece);
                } else if part["type"].as_str() == Some("text") {
                    return Err("text block requires text".into());
                }
            }
            Ok(text)
        }
        _ => Err("system must be text or text blocks".into()),
    }
}

fn append_system(messages: &mut Vec<Value>, content: &Value) -> Result<(), String> {
    let text = anthropic_text(content)?;
    if !text.is_empty() {
        messages.push(json!({"role":"system","content":text}));
    }
    Ok(())
}

fn append_tool(messages: &mut Vec<Value>, message: &Value) -> Result<(), String> {
    let call_id = message["tool_use_id"]
        .as_str()
        .or_else(|| message["tool_call_id"].as_str())
        .ok_or("tool message requires tool_use_id")?;
    messages.push(json!({
        "role": "tool",
        "tool_call_id": call_id,
        "content": tool_result_content(message)?
    }));
    Ok(())
}

fn anthropic_messages(body: &Value) -> Result<Vec<Value>, String> {
    let mut messages = Vec::new();
    if let Some(system) = body.get("system") {
        append_system(&mut messages, system)?;
    }
    let incoming = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("messages must be an array")?;
    for message in incoming {
        match message["role"].as_str() {
            Some("user") => append_user(&mut messages, &message["content"])?,
            Some("assistant") => messages.push(assistant_message(&message["content"])?),
            Some("system" | "developer") => append_system(&mut messages, &message["content"])?,
            Some("tool") => append_tool(&mut messages, message)?,
            other => {
                return Err(format!(
                    "message role must be user or assistant{}",
                    other.map(|role| format!(", not {role}")).unwrap_or_default()
                ));
            }
        }
    }
    Ok(messages)
}

fn append_user(messages: &mut Vec<Value>, content: &Value) -> Result<(), String> {
    if let Some(text) = content.as_str() {
        messages.push(json!({"role":"user","content":text}));
        return Ok(());
    }
    let parts = content
        .as_array()
        .ok_or("user content must be text or content blocks")?;
    let mut pending = Vec::new();
    for part in parts {
        match part["type"].as_str() {
            Some("text") => pending.push(json!({
                "type": "text",
                "text": part["text"].as_str().ok_or("text block requires text")?
            })),
            Some("image") => pending.push(image_part(part)?),
            Some("tool_result") => {
                flush_user(messages, &mut pending);
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": part["tool_use_id"].as_str().ok_or("tool_result requires tool_use_id")?,
                    "content": tool_result_content(part)?
                }));
            }
            Some("thinking" | "redacted_thinking") => {}
            other => {
                return Err(format!(
                    "unsupported user content block: {}",
                    other.unwrap_or("missing type")
                ));
            }
        }
    }
    flush_user(messages, &mut pending);
    Ok(())
}

fn flush_user(messages: &mut Vec<Value>, pending: &mut Vec<Value>) {
    if pending.is_empty() {
        return;
    }
    messages.push(json!({"role":"user","content":std::mem::take(pending)}));
}

fn image_part(part: &Value) -> Result<Value, String> {
    let source = part.get("source").ok_or("image block requires source")?;
    match source["type"].as_str() {
        Some("base64") => {
            let media_type = source["media_type"]
                .as_str()
                .ok_or("image source requires media_type")?;
            let data = source["data"].as_str().ok_or("image source requires data")?;
            Ok(json!({"type":"image_url","image_url":{"url":format!("data:{media_type};base64,{data}")}}))
        }
        Some("url") => Err("images must be inline data URLs; remote URLs are not fetched".into()),
        _ => Err("image source type must be base64".into()),
    }
}

fn tool_result_content(part: &Value) -> Result<Value, String> {
    match &part["content"] {
        Value::Null => Ok(json!("")),
        Value::String(text) => Ok(json!(text)),
        Value::Array(parts) => {
            let mut output = Vec::new();
            for item in parts {
                match item["type"].as_str() {
                    Some("text") | None if item.get("text").is_some() => output.push(json!({
                        "type": "text",
                        "text": item["text"].as_str().ok_or("text block requires text")?
                    })),
                    Some("image") => output.push(image_part(item)?),
                    other => {
                        return Err(format!(
                            "unsupported tool_result block: {}",
                            other.unwrap_or("missing type")
                        ));
                    }
                }
            }
            Ok(Value::Array(output))
        }
        _ => Err("tool_result content must be text or content blocks".into()),
    }
}

fn assistant_message(content: &Value) -> Result<Value, String> {
    if let Some(text) = content.as_str() {
        return Ok(json!({"role":"assistant","content":text}));
    }
    let parts = content
        .as_array()
        .ok_or("assistant content must be text or content blocks")?;
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for part in parts {
        match part["type"].as_str() {
            Some("text") => text.push_str(part["text"].as_str().ok_or("text block requires text")?),
            Some("thinking") => reasoning.push_str(
                part["thinking"]
                    .as_str()
                    .ok_or("thinking block requires thinking")?,
            ),
            Some("redacted_thinking") => {}
            Some("tool_use") => {
                let input = part.get("input").cloned().unwrap_or(json!({}));
                let arguments = if let Some(raw) = input.as_str() {
                    raw.to_string()
                } else {
                    input.to_string()
                };
                tool_calls.push(json!({
                    "id": part["id"].as_str().ok_or("tool_use requires id")?,
                    "type": "function",
                    "function": {
                        "name": part["name"].as_str().ok_or("tool_use requires name")?,
                        "arguments": arguments
                    }
                }));
            }
            other => {
                return Err(format!(
                    "unsupported assistant content block: {}",
                    other.unwrap_or("missing type")
                ));
            }
        }
    }
    let mut message = json!({"role":"assistant","content":text});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }
    Ok(message)
}

fn estimate_tokens(request: &Value) -> u64 {
    let mut tokens = 0;
    if let Some(messages) = request["messages"].as_array() {
        for message in messages {
            tokens += estimate_value(&message["content"]);
            if let Some(reasoning) = message["reasoning_content"].as_str() {
                tokens += estimate_text(reasoning);
            }
            if let Some(calls) = message["tool_calls"].as_array() {
                tokens += estimate_text(&Value::Array(calls.clone()).to_string());
            }
        }
    }
    if let Some(tools) = request["tools"].as_array() {
        tokens += estimate_text(&Value::Array(tools.clone()).to_string());
    }
    if let Some(images) = request.get("images").and_then(Value::as_object) {
        tokens += images.len() as u64 * 2048;
    }
    tokens.max(1)
}

fn estimate_value(value: &Value) -> u64 {
    match value {
        Value::String(text) => estimate_text(text),
        Value::Array(parts) => parts.iter().map(estimate_value).sum(),
        Value::Object(object) if object.get("type").and_then(Value::as_str) == Some("text") => {
            estimate_text(object.get("text").and_then(Value::as_str).unwrap_or(""))
        }
        Value::Object(object) if object.get("type").and_then(Value::as_str) == Some("image") => 2048,
        other => estimate_text(&other.to_string()),
    }
}

fn estimate_text(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    (chars + 1) / 2
}

pub struct AnthropicStream {
    id: String,
    model: String,
    index: usize,
    thinking_open: bool,
    text_open: bool,
    emitted_thinking: bool,
    emitted_text: bool,
}

impl AnthropicStream {
    pub fn new(id: &str, model: &str) -> Self {
        Self {
            id: id.into(),
            model: model.into(),
            index: 0,
            thinking_open: false,
            text_open: false,
            emitted_thinking: false,
            emitted_text: false,
        }
    }

    pub fn started(&self, input_tokens: u64) -> Value {
        json!({
            "type": "message_start",
            "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0
                }
            }
        })
    }

    pub fn delta(&mut self, reasoning: bool, text: &Value) -> Vec<Value> {
        let mut events = Vec::new();
        if reasoning {
            self.ensure_thinking(&mut events);
            events.push(json!({
                "type": "content_block_delta",
                "index": self.index,
                "delta": {"type": "thinking_delta", "thinking": text}
            }));
        } else {
            self.close_thinking(&mut events);
            self.ensure_text(&mut events);
            events.push(json!({
                "type": "content_block_delta",
                "index": self.index,
                "delta": {"type": "text_delta", "text": text}
            }));
        }
        events
    }

    pub fn finish(&mut self, message: &Value) -> Vec<Value> {
        let mut events = Vec::new();
        self.close_open(&mut events);
        if let Some(blocks) = message["content"].as_array() {
            for block in blocks {
                match block["type"].as_str() {
                    Some("thinking") if !self.emitted_thinking => {
                        self.emit_block(&mut events, block);
                    }
                    Some("text") if !self.emitted_text => self.emit_block(&mut events, block),
                    Some("tool_use") => self.emit_tool(&mut events, block),
                    _ => {}
                }
            }
        }
        events.push(json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": message["stop_reason"],
                "stop_sequence": Value::Null
            },
            "usage": {"output_tokens": message["usage"]["output_tokens"]}
        }));
        events.push(json!({"type":"message_stop"}));
        events
    }

    fn ensure_thinking(&mut self, events: &mut Vec<Value>) {
        if self.thinking_open {
            return;
        }
        events.push(json!({
            "type": "content_block_start",
            "index": self.index,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""}
        }));
        self.thinking_open = true;
        self.emitted_thinking = true;
    }

    fn ensure_text(&mut self, events: &mut Vec<Value>) {
        if self.text_open {
            return;
        }
        events.push(json!({
            "type": "content_block_start",
            "index": self.index,
            "content_block": {"type": "text", "text": ""}
        }));
        self.text_open = true;
        self.emitted_text = true;
    }

    fn close_thinking(&mut self, events: &mut Vec<Value>) {
        if self.thinking_open {
            events.push(json!({"type":"content_block_stop","index":self.index}));
            self.index += 1;
            self.thinking_open = false;
        }
    }

    fn close_text(&mut self, events: &mut Vec<Value>) {
        if self.text_open {
            events.push(json!({"type":"content_block_stop","index":self.index}));
            self.index += 1;
            self.text_open = false;
        }
    }

    fn close_open(&mut self, events: &mut Vec<Value>) {
        self.close_thinking(events);
        self.close_text(events);
    }

    fn emit_block(&mut self, events: &mut Vec<Value>, block: &Value) {
        events.push(json!({
            "type": "content_block_start",
            "index": self.index,
            "content_block": block
        }));
        events.push(json!({"type":"content_block_stop","index":self.index}));
        self.index += 1;
        if block["type"] == "thinking" {
            self.emitted_thinking = true;
        }
        if block["type"] == "text" {
            self.emitted_text = true;
        }
    }

    fn emit_tool(&mut self, events: &mut Vec<Value>, block: &Value) {
        let arguments = block["input"].to_string();
        events.push(json!({
            "type": "content_block_start",
            "index": self.index,
            "content_block": {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": {}
            }
        }));
        events.push(json!({
            "type": "content_block_delta",
            "index": self.index,
            "delta": {"type": "input_json_delta", "partial_json": arguments}
        }));
        events.push(json!({"type":"content_block_stop","index":self.index}));
        self.index += 1;
    }
}
