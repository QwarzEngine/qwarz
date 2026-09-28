//! JSONL control protocol shared with the Rust supervisor.
//!
//! One generation is active at a time. This build accepts the session and
//! answers `engine_not_linked`: the CUDA kernels are not in the binary yet,
//! and ExLlamaV3 remains the serving worker.

use serde_json::{Value, json};

const MAX_LINE_BYTES: usize = 32 * 1024 * 1024;
const PROTOCOL: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Continue,
    Stop,
}

pub struct Worker {
    active: Option<String>,
}

impl Worker {
    pub fn new() -> Self {
        Self { active: None }
    }

    pub fn ready(&self) -> Value {
        self.ready_with(json!({
            "model": "qwasar-qwen38-27b",
            "context_size": crate::profile::CONTEXT_TOKENS,
            "vision": true,
            "streams": 1,
            "engine": "q38",
            "linked": false,
            "draft_method": "mtp",
            "native_context_size": crate::profile::CONTEXT_TOKENS
        }))
    }

    pub fn ready_with(&self, config: Value) -> Value {
        json!({"type": "ready", "protocol": PROTOCOL, "config": config})
    }

    pub fn push(&mut self, line: &str) -> (Step, Vec<Value>) {
        if line.len() > MAX_LINE_BYTES {
            return (Step::Stop, vec![terminal("", "invalid_request", "worker protocol line exceeds 32 MiB", 400)]);
        }
        let payload = match serde_json::from_str::<Value>(line) {
            Ok(value) if value.is_object() => value,
            Ok(_) => return (Step::Continue, vec![terminal("", "invalid_request", "protocol command must be an object", 400)]),
            Err(_) => return (Step::Continue, vec![terminal("", "invalid_request", "protocol command is not JSON", 400)]),
        };
        let id = match payload.get("id") {
            Some(Value::String(id)) => id.clone(),
            Some(_) => return (Step::Continue, vec![terminal("", "invalid_request", "request id must be a string", 400)]),
            None => String::new(),
        };
        match payload.get("op").and_then(Value::as_str) {
            Some("shutdown") => {
                self.active = None;
                (Step::Stop, Vec::new())
            }
            Some("cancel") => {
                if self.active.as_deref() == Some(id.as_str()) {
                    self.active = None;
                }
                (Step::Continue, Vec::new())
            }
            Some("generate") if id.is_empty() || id.len() > 256 => {
                (Step::Continue, vec![terminal(&id, "invalid_request", "generate needs a nonempty id of at most 256 characters", 400)])
            }
            Some("generate") if self.active.is_some() => {
                (Step::Continue, vec![terminal(&id, "runtime_busy", "one generation is already active", 409)])
            }
            Some("generate") => {
                self.active = Some(id.clone());
                let event = terminal(
                    &id,
                    "engine_not_linked",
                    "q38 worker has no CUDA kernels yet; ExLlamaV3 remains the serving engine",
                    503,
                );
                self.active = None;
                (Step::Continue, vec![event])
            }
            _ => (Step::Continue, vec![terminal(&id, "invalid_request", "generate needs a nonempty id of at most 256 characters", 400)]),
        }
    }
}

impl Default for Worker {
    fn default() -> Self {
        Self::new()
    }
}

fn terminal(id: &str, code: &str, message: &str, http_status: u16) -> Value {
    json!({
        "type": "terminal",
        "id": id,
        "status": "failed",
        "message": {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []},
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0}},
        "metrics": {},
        "snapshot": null,
        "error": {"code": code, "message": message, "http_status": http_status}
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_advertises_one_stream_full_context_and_vision() {
        let ready = Worker::new().ready();
        assert_eq!(ready["protocol"], 1);
        assert_eq!(ready["config"]["context_size"], 262_144);
        assert_eq!(ready["config"]["vision"], true);
        assert_eq!(ready["config"]["streams"], 1);
        assert_eq!(ready["config"]["linked"], false);
    }

    #[test]
    fn a_second_generation_is_rejected_while_one_is_active() {
        let mut worker = Worker::new();
        worker.active = Some("resp_a".into());
        let (step, events) = worker.push(r#"{"op":"generate","id":"resp_b","request":{}}"#);
        assert_eq!(step, Step::Continue);
        assert_eq!(events[0]["error"]["code"], "runtime_busy");
        assert_eq!(events[0]["error"]["http_status"], 409);
        assert_eq!(worker.active.as_deref(), Some("resp_a"));
    }

    #[test]
    fn generate_fails_closed_until_kernels_exist() {
        let mut worker = Worker::new();
        let (step, events) = worker.push(r#"{"op":"generate","id":"resp_a","request":{}}"#);
        assert_eq!(step, Step::Continue);
        assert_eq!(events[0]["error"]["code"], "engine_not_linked");
        assert!(worker.active.is_none());
        let (_, shutdown) = worker.push(r#"{"op":"shutdown"}"#);
        assert!(shutdown.is_empty());
    }

    #[test]
    fn malformed_commands_stay_on_the_stream() {
        let mut worker = Worker::new();
        let (_, events) = worker.push("[]");
        assert_eq!(events[0]["error"]["message"], "protocol command must be an object");
        let (_, events) = worker.push(r#"{"op":"generate","id":""}"#);
        assert_eq!(events[0]["error"]["code"], "invalid_request");
        let huge = "x".repeat(32 * 1024 * 1024 + 1);
        let (step, _) = worker.push(&huge);
        assert_eq!(step, Step::Stop);
    }
}
