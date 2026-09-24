use qwasar_server::{anthropic, api, store::Store};
use serde_json::json;

#[test]
fn pi_request_preserves_tools_and_normalizes_text_parts() {
    let request = api::prepare(&json!({"model":"qwasar-qwen38-27b","messages":[
        {"role":"system","content":"coding"},
        {"role":"user","content":[{"type":"text","text":"hello"}]}],
        "stream":true,"stream_options":{"include_usage":true},
        "chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":true},
        "tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    }), false, None).unwrap();
    assert_eq!(request["thinking"], "off");
    assert_eq!(request["messages"][1]["content"], "hello");
    assert_eq!(request["tools"][0]["function"]["name"], "read");
}

#[test]
fn rejects_unsupported_fields_and_bad_sampling_without_dispatch() {
    for additional in [
        json!({"temperature":-1}),
        json!({"max_tokens":0}),
        json!({"frequency_penalty":1}),
        json!({"stream":"yes"}),
        json!({"chat_template_kwargs":{"preserve_thinking":false}}),
    ] {
        let mut body =
            json!({"model":"qwasar-qwen38-27b","messages":[{"role":"user","content":"hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(additional.as_object().unwrap().clone());
        assert!(api::prepare(&body, false, None).is_err(), "{body}");
    }
}

#[test]
fn responses_do_not_inherit_instructions_and_validate_function_results() {
    let parent = json!({"messages":[{"role":"system","content":"old instructions"},
        {"role":"user","content":"read"},
        {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]}]});
    let request = api::prepare(
        &json!({"model":"qwasar-qwen38-27b","previous_response_id":"resp_old",
        "input":[{"type":"function_call_output","call_id":"call_1","output":"file data"}]}),
        true,
        Some(&parent),
    )
    .unwrap();
    assert!(
        request["messages"]
            .as_array()
            .unwrap()
            .iter()
            .all(|message| message["role"] != "system")
    );
    assert_eq!(request["messages"][2]["tool_call_id"], "call_1");
    let invalid = json!({"model":"qwasar-qwen38-27b","input":[{"type":"function_call_output","call_id":"call_missing","output":"bad"}]});
    assert!(api::prepare(&invalid, true, Some(&parent)).is_err());
}

#[test]
fn store_only_commits_completed_snapshots_and_survives_reopen() {
    let path = std::env::temp_dir().join(format!(
        "qwasar-store-{}-{}.db",
        std::process::id(),
        api::new_id("test")
    ));
    {
        let store = Store::open(&path).unwrap();
        store
            .begin("resp_one", &json!({"messages":[]}), Some("key_one"))
            .unwrap();
        assert!(store.parent("resp_one").is_err());
        let snapshot = json!({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"answer","reasoning_content":"hidden"}],"tape":[1,2,3]});
        store
            .finish(
                "resp_one",
                "completed",
                &json!({"id":"resp_one","status":"completed"}),
                Some(&snapshot),
            )
            .unwrap();
        assert_eq!(store.parent("resp_one").unwrap()["tape"], json!([1, 2, 3]));
        let messages = json!([{"role":"system","content":"changed"},{"role":"user","content":"hi"},{"role":"assistant","content":"answer"},{"role":"user","content":"next"}]);
        assert_eq!(
            store
                .match_parent(messages.as_array().unwrap())
                .unwrap()
                .unwrap()["tape"],
            json!([1, 2, 3])
        );
        store.begin("resp_stale", &json!({}), None).unwrap();
    }
    let store = Store::open(&path).unwrap();
    assert_eq!(store.parent("resp_one").unwrap()["tape"], json!([1, 2, 3]));
    assert!(store.parent("resp_stale").is_err());
    assert_eq!(store.get("resp_stale").unwrap()["status"], "failed");
    assert_eq!(store.idempotent("key_one").unwrap().unwrap().0, "resp_one");
    drop(store);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn wire_completion_distinguishes_tools_length_and_reasoning() {
    let terminal = json!({"status":"completed","message":{"role":"assistant","content":"","reasoning_content":"thought","tool_calls":[{"id":"call_a","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]},"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}});
    let response = api::chat_result("resp_a", &terminal);
    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        response["choices"][0]["message"]["reasoning_content"],
        "thought"
    );
    let mut truncated = terminal.clone();
    truncated["status"] = json!("incomplete");
    truncated["metrics"] = json!({"incomplete_reason":"max_new_tokens","finish_reason":"max_new_tokens"});
    assert_eq!(
        api::chat_result("resp_b", &truncated)["choices"][0]["finish_reason"],
        "length"
    );
    let mut rejected = terminal.clone();
    rejected["status"] = json!("incomplete");
    rejected["metrics"] = json!({"incomplete_reason":"undeclared_tool","finish_reason":"stop_token",
        "tool_error":{"stage":"native_parse","tool":"browser","error":"undeclared or prohibited tool"}});
    let chat = api::chat_result("resp_c", &rejected);
    assert_eq!(chat["choices"][0]["finish_reason"], "stop");
    assert_eq!(chat["qwasar_metrics"]["tool_error"]["tool"], "browser");
    let response = api::response_result("resp_c", &rejected);
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["incomplete_details"], serde_json::Value::Null);
    let response = api::response_result("resp_b", &truncated);
    assert_eq!(response["incomplete_details"]["reason"], "max_output_tokens");
    let response = api::response_result("resp_a", &terminal);
    assert_eq!(response["output"][1]["type"], "function_call");
    assert_eq!(response["output"][1]["call_id"], "call_a");
    assert_eq!(anthropic::stop_reason(&rejected), "end_turn");
    assert_eq!(anthropic::stop_reason(&truncated), "max_tokens");
}

#[test]
fn omitted_output_limit_is_native_maximum() {
    let responses = api::prepare(
        &json!({"model":api::MODEL,"input":"hello","reasoning":{"effort":"high"}}),
        true,
        None,
    )
    .unwrap();
    assert_eq!(responses["max_tokens"], 32768);
    assert_eq!(responses["thinking"], "xhigh");
    let chat = api::prepare(
        &json!({"model":api::MODEL,"messages":[{"role":"user","content":"hello"}]}),
        false,
        None,
    )
    .unwrap();
    assert_eq!(chat["max_tokens"], 32768);
}

#[test]
fn qwen_chat_template_reasoning_effort_is_accepted() {
    let request = api::prepare(
        &json!({
            "model": api::MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {
                "enable_thinking": true,
                "reasoning_effort": "xhigh"
            }
        }),
        false,
        None,
    )
    .unwrap();
    assert_eq!(request["thinking"], "xhigh");
    let high = api::prepare(
        &json!({
            "model": api::MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "high"}
        }),
        false,
        None,
    )
    .unwrap();
    assert_eq!(high["thinking"], "xhigh");
    for additional in [
        json!({"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"xhigh","foo":1}}),
        json!({"reasoning_effort":"low","chat_template_kwargs":{"reasoning_effort":"xhigh"}}),
    ] {
        let mut body = json!({"model":api::MODEL,"messages":[{"role":"user","content":"hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(additional.as_object().unwrap().clone());
        assert!(api::prepare(&body, false, None).is_err(), "{body}");
    }
}

#[test]
fn hermes_title_generation_aliases_are_accepted() {
    let title = api::prepare(
        &json!({
            "model": api::MODEL,
            "messages": [{"role": "user", "content": "Build a crayon Tetris"}],
            "reasoning_effort": "none",
            "max_tokens": 64,
            "temperature": 0.3,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "session_title",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                        "additionalProperties": false
                    }
                }
            }
        }),
        false,
        None,
    )
    .unwrap();
    assert_eq!(title["thinking"], "off");
    assert_eq!(title["max_tokens"], 64);
    let disabled = api::prepare(
        &json!({
            "model": api::MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning": {"enabled": false}
        }),
        false,
        None,
    )
    .unwrap();
    assert_eq!(disabled["thinking"], "off");
    let ultra = api::prepare(
        &json!({
            "model": api::MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "ultra"
        }),
        false,
        None,
    )
    .unwrap();
    assert_eq!(ultra["thinking"], "xhigh");
}

#[test]
fn reasoning_budget_tokens_is_passed_through() {
    let request = api::prepare(
        &json!({"model":api::MODEL,"messages":[{"role":"user","content":"hello"}],
            "reasoning_budget_tokens":2048}),
        false,
        None,
    )
    .unwrap();
    assert_eq!(request["reasoning_budget_tokens"], 2048);
    let disabled = api::prepare(
        &json!({"model":api::MODEL,"messages":[{"role":"user","content":"hello"}],
            "reasoning_budget_tokens":0}),
        false,
        None,
    )
    .unwrap();
    assert_eq!(disabled["reasoning_budget_tokens"], 0);
    for additional in [
        json!({"reasoning_budget_tokens":-1}),
        json!({"reasoning_budget_tokens":32769}),
        json!({"reasoning_budget_tokens":"8192"}),
    ] {
        let mut body = json!({"model":api::MODEL,"messages":[{"role":"user","content":"hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(additional.as_object().unwrap().clone());
        assert!(api::prepare(&body, false, None).is_err(), "{body}");
    }
}

#[test]
fn ignored_controls_are_rejected_instead_of_silently_defaulted() {
    for additional in [
        json!({"reasoning_effort":false}),
        json!({"reasoning":"medium"}),
        json!({"reasoning_effort":"low","reasoning":{"effort":"high"}}),
        json!({"store":"yes"}),
    ] {
        let mut body = json!({"model":api::MODEL,"messages":[{"role":"user","content":"hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(additional.as_object().unwrap().clone());
        assert!(
            api::prepare(&body, false, None).is_err(),
            "accepted ignored control: {body}"
        );
    }
}

#[test]
fn codex_responses_extras_are_accepted_without_changing_generation() {
    let request = api::prepare(
        &json!({
            "model":"qwasar-qwen38-27b",
            "instructions":"You are Codex.",
            "input":[
                {"type":"message","role":"developer","content":[{"type":"input_text","text":"Use local tools."}]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},
                {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"prior"}]}
            ],
            "stream":true,
            "store":false,
            "generate":false,
            "parallel_tool_calls":false,
            "metadata":{"source":"codex"},
            "include":["reasoning.encrypted_content"],
            "prompt_cache_key":"session",
            "text":{"verbosity":"medium"},
            "client_metadata":{"session_id":"s","thread_id":"t"},
            "service_tier":"default",
            "access_programs":{"cyber":"standard"},
            "reasoning":{"effort":"high","summary":"auto","context":"current_turn"},
            "stream_options":{"include_usage":true,"reasoning_summary_delivery":"sequential_cutoff"},
            "tools":[
                {"type":"function","name":"exec_command","description":"Run a command.","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}},
                {"type":"custom","name":"apply_patch","description":"Edit files.","format":{"type":"grammar","syntax":"lark","definition":"start: patch"}},
                {"type":"local_shell"},
                {"type":"web_search","external_web_access":true}
            ],
            "max_output_tokens":64
        }),
        true,
        None,
    )
    .unwrap();
    assert_eq!(request["thinking"], "xhigh");
    assert_eq!(request["messages"][0]["role"], "system");
    assert_eq!(
        request["messages"][0]["content"],
        "You are Codex.\n\nUse local tools."
    );
    assert_eq!(request["messages"][1]["content"], "hello");
    assert_eq!(request["messages"].as_array().unwrap().len(), 2);
    assert_eq!(request["tools"][0]["function"]["name"], "exec_command");
    assert_eq!(request["tools"][1]["function"]["name"], "apply_patch");
    assert_eq!(request["tools"][2]["function"]["name"], "local_shell");
    assert_eq!(request["tools"][3]["function"]["name"], "web_search");
}

#[test]
fn models_payload_keeps_openai_list_and_codex_catalog() {
    let payload = api::models_payload();
    assert_eq!(payload["data"][0]["id"], api::MODEL);
    assert_eq!(payload["models"][0]["slug"], api::MODEL);
    assert_eq!(payload["models"][0]["context_window"], 262144);
    assert_eq!(payload["models"][0]["supported_reasoning_levels"][3]["effort"], "xhigh");
    assert_eq!(payload["models"][0]["input_modalities"], json!(["text", "image"]));
    assert_eq!(payload["models"][0]["supports_image_detail_original"], false);
    assert!(payload["models"][0]["base_instructions"].as_str().unwrap().contains("Codex"));
}

#[test]
fn history_matching_honors_supplied_reasoning_and_rejects_ambiguous_omission() {
    let store = Store::open(std::path::Path::new(":memory:")).unwrap();
    for (id, thought, tape) in [("older", "first thought", 1), ("newer", "other thought", 2)] {
        store.begin(id, &json!({}), None).unwrap();
        store
            .finish(
                id,
                "completed",
                &json!({}),
                Some(&json!({"messages":[
            {"role":"user","content":"hello"},
            {"role":"assistant","content":"same answer","reasoning_content":thought}
        ],"tape":[tape]})),
            )
            .unwrap();
    }
    let mut messages = vec![
        json!({"role":"user","content":"hello"}),
        json!({"role":"assistant","content":"same answer","reasoning_content":"first thought"}),
        json!({"role":"user","content":"next"}),
    ];
    assert_eq!(
        store.match_parent(&messages).unwrap().unwrap()["tape"],
        json!([1])
    );
    messages[1]
        .as_object_mut()
        .unwrap()
        .remove("reasoning_content");
    assert!(
        store.match_parent(&messages).is_err(),
        "ambiguous exact histories must not pick newest"
    );
    messages[1]["reasoning_content"] = json!("edited thought");
    assert!(store.match_parent(&messages).unwrap().is_none());
}

#[test]
fn terminal_timestamp_is_identical_in_committed_and_returned_views() {
    let terminal = json!({"created_at":12345,"status":"completed","message":{"role":"assistant","content":"answer"}});
    assert_eq!(api::response_result("same", &terminal)["created_at"], 12345);
    assert_eq!(api::chat_result("same", &terminal)["created"], 12345);
}

const TINY_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

#[test]
fn inline_images_become_hash_parts_attached_to_the_request() {
    let url = format!("data:image/png;base64,{TINY_PNG}");
    let request = api::prepare(&json!({"model":"qwasar-qwen38-27b","messages":[
        {"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"},
            {"type":"image_url","image_url":{"url":url}},{"type":"text","text":"c"}]}]}), false, None).unwrap();
    let content = request["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 3);
    assert_eq!(content[0], json!({"type":"text","text":"ab"}));
    assert_eq!(content[1]["type"], "image");
    let sha256 = content[1]["sha256"].as_str().unwrap();
    assert_eq!(sha256.len(), 64);
    assert_eq!(request["images"][sha256]["media_type"], "image/png");
    assert_eq!(request["images"][sha256]["data"], TINY_PNG);
    // Responses input_image and hash-only parts from a parent snapshot are accepted; data stays null.
    let parent = json!({"messages":[{"role":"user","content":[{"type":"image","sha256":sha256,"media_type":"image/png"}]},
        {"role":"assistant","content":"seen"}]});
    let request = api::prepare(&json!({"model":"qwasar-qwen38-27b","previous_response_id":"resp_old",
        "input":[{"role":"user","content":[{"type":"input_image","image_url":url}]}]}), true, Some(&parent)).unwrap();
    assert_eq!(request["messages"][0]["content"][0]["sha256"], sha256);
    assert_eq!(request["images"][sha256]["data"], TINY_PNG);
    let hash_only = api::prepare(&json!({"model":"qwasar-qwen38-27b","previous_response_id":"resp_old","input":"again"}), true, Some(&parent)).unwrap();
    assert!(hash_only["images"][sha256]["data"].is_null());
    let codex = api::prepare(&json!({"model":"qwasar-qwen38-27b","input":[
        {"type":"input_text","text":"look"},
        {"type":"input_image","image_url":{"url":url},"detail":"high"}]}), true, None).unwrap();
    assert_eq!(codex["messages"][0]["content"][0]["text"], "look");
    assert_eq!(codex["messages"][0]["content"][1]["sha256"], sha256);
    // Stored bodies carry hash references instead of the inline payload.
    let redacted = api::redact_images(&json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":url}}]}]}));
    assert_eq!(redacted["messages"][0]["content"][0]["sha256"], sha256);
}

#[test]
fn images_are_rejected_outside_user_messages_or_without_inline_data() {
    let url = format!("data:image/png;base64,{TINY_PNG}");
    let image = json!({"type":"image_url","image_url":{"url":url}});
    for messages in [
        json!([{"role":"system","content":[image]},{"role":"user","content":"hi"}]),
        json!([{"role":"user","content":"hi"},{"role":"assistant","content":[image]},{"role":"user","content":"x"}]),
        json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]),
        json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/bmp;base64,AAAA"}}]}]),
        json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,!!!"}}]}]),
        json!([{"role":"user","content":[{"type":"image","sha256":"ABC","media_type":"image/png"}]}]),
    ] {
        let body = json!({"model":"qwasar-qwen38-27b","messages":messages});
        assert!(api::prepare(&body, false, None).is_err(), "{body}");
    }
    let many_same: Vec<_> = (0..=api::MAX_IMAGES).map(|_| image.clone()).collect();
    let request = api::prepare(&json!({"model":"qwasar-qwen38-27b","messages":[{"role":"user","content":many_same}]}), false, None).unwrap();
    assert_eq!(request["images"].as_object().unwrap().len(), 1);
    let mut parts = Vec::new();
    for index in 0..=api::MAX_IMAGES {
        parts.push(json!({"type":"image","sha256":format!("{index:064x}"),"media_type":"image/png"}));
    }
    let parent = json!({"messages":[{"role":"user","content":parts},{"role":"assistant","content":"seen"}]});
    let pruned = api::prepare(&json!({"model":"qwasar-qwen38-27b","previous_response_id":"resp_old","input":"again"}), true, Some(&parent)).unwrap();
    let kept = pruned["messages"][0]["content"].as_array().unwrap();
    assert_eq!(kept.len(), api::MAX_IMAGES);
    assert_eq!(kept[0]["sha256"], format!("{:064x}", 1));
    assert_eq!(kept[api::MAX_IMAGES - 1]["sha256"], format!("{:064x}", api::MAX_IMAGES));
}

#[test]
fn tool_result_images_are_hoisted_onto_a_following_user_message() {
    let url = format!("data:image/png;base64,{TINY_PNG}");
    let request = api::prepare(
        &json!({
            "model": "qwasar-qwen38-27b",
            "messages": [
                {"role": "user", "content": "look at this sketch"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_vision",
                    "type": "function",
                    "function": {"name": "vision_analyze", "arguments": "{\"question\":\"style\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_vision", "content": [
                    {"type": "text", "text": "Image loaded into your context"},
                    {"type": "image_url", "image_url": {"url": url}}
                ]}
            ]
        }),
        false,
        None,
    )
    .unwrap();
    assert_eq!(request["messages"][2]["role"], "tool");
    assert_eq!(request["messages"][2]["content"], "Image loaded into your context");
    assert_eq!(request["messages"][3]["role"], "user");
    let content = request["messages"][3]["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "image");
    let sha256 = content[0]["sha256"].as_str().unwrap();
    assert_eq!(request["images"][sha256]["data"], TINY_PNG);
}

#[test]
fn codex_view_image_output_is_hoisted_onto_a_following_user_message() {
    let url = format!("data:image/png;base64,{TINY_PNG}");
    let parent = json!({"messages":[
        {"role":"user","content":"use reference.jpg"},
        {"role":"assistant","content":"","tool_calls":[{
            "id":"call_view",
            "type":"function",
            "function":{"name":"view_image","arguments":"{\"path\":\"reference.jpg\"}"}
        }]}
    ]});
    for output in [
        json!([{"type":"input_image","image_url":url}]),
        json!([{"type":"input_text","text":"Image loaded into your context"},
            {"type":"input_image","image_url":{"url":url},"detail":"auto"}]),
        json!({"type":"input_image","image_url":url}),
    ] {
        let request = api::prepare(
            &json!({
                "model":"qwasar-qwen38-27b",
                "previous_response_id":"resp_old",
                "input":[{"type":"function_call_output","call_id":"call_view","output":output}]
            }),
            true,
            Some(&parent),
        )
        .unwrap();
        assert_eq!(request["messages"][2]["role"], "tool");
        assert_eq!(request["messages"][2]["tool_call_id"], "call_view");
        assert_eq!(request["messages"][3]["role"], "user");
        let content = request["messages"][3]["content"].as_array().unwrap();
        assert_eq!(content.last().unwrap()["type"], "image");
        let sha256 = content.last().unwrap()["sha256"].as_str().unwrap();
        assert_eq!(request["images"][sha256]["data"], TINY_PNG);
    }
    let same_turn = api::prepare(
        &json!({
            "model":"qwasar-qwen38-27b",
            "input":[
                {"type":"input_text","text":"use this sketch"},
                {"type":"function_call","call_id":"call_view","name":"view_image",
                    "arguments":"{\"path\":\"reference.jpg\"}"},
                {"type":"function_call_output","call_id":"call_view",
                    "output":[{"type":"input_image","image_url":url}]}
            ]
        }),
        true,
        None,
    )
    .unwrap();
    assert_eq!(same_turn["messages"][2]["role"], "tool");
    assert_eq!(same_turn["messages"][3]["content"][0]["type"], "image");
}

#[test]
fn codex_parallel_view_image_waits_for_sibling_tool_results() {
    let url = format!("data:image/png;base64,{TINY_PNG}");
    for outputs in [
        json!([
            {"type":"function_call_output","call_id":"call_exec","output":"prompt text"},
            {"type":"function_call_output","call_id":"call_view",
                "output":[{"type":"input_image","image_url":url}]}
        ]),
        json!([
            {"type":"function_call_output","call_id":"call_view",
                "output":[{"type":"input_image","image_url":url}]},
            {"type":"function_call_output","call_id":"call_exec","output":"prompt text"}
        ]),
    ] {
        let mut input = json!([
            {"type":"input_text","text":"use this sketch"},
            {"type":"function_call","call_id":"call_exec","name":"exec_command",
                "arguments":"{\"cmd\":\"cat input.md\"}"},
            {"type":"function_call","call_id":"call_view","name":"view_image",
                "arguments":"{\"path\":\"reference.jpg\"}"}
        ]);
        input.as_array_mut().unwrap().extend(outputs.as_array().unwrap().iter().cloned());
        let request = api::prepare(
            &json!({"model":"qwasar-qwen38-27b","input":input}),
            true,
            None,
        )
        .unwrap();
        assert_eq!(request["messages"][1]["role"], "assistant");
        assert_eq!(request["messages"][1]["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(request["messages"][2]["role"], "tool");
        assert_eq!(request["messages"][3]["role"], "tool");
        assert_eq!(request["messages"][4]["role"], "user");
        assert_eq!(request["messages"][4]["content"][0]["type"], "image");
        let sha256 = request["messages"][4]["content"][0]["sha256"].as_str().unwrap();
        assert_eq!(request["images"][sha256]["data"], TINY_PNG);
    }
}

#[test]
fn droid_parallel_image_results_are_repaired_after_interleaved_hoists() {
    let url = format!("data:image/png;base64,{TINY_PNG}");
    let hoist = json!([
        {"type": "text", "text": "Image content from tool result:"},
        {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
    ]);
    let request = api::prepare(
        &json!({
            "model": "qwasar-qwen38-27b",
            "messages": [
                {"role": "user", "content": "look at the screenshots"},
                {"role": "assistant", "content": "viewing", "tool_calls": [{
                    "id": "call_f0a4339686c323854b75eb66",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{\"file_path\":\"/tmp/a.png\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_f0a4339686c323854b75eb66",
                    "content": "Image file: a.png"},
                {"role": "user", "content": hoist.clone()},
                {"role": "tool", "tool_call_id": "call_85ee193b4a64700c420cadc4",
                    "content": "Image file: b.png"},
                {"role": "user", "content": hoist}
            ]
        }),
        false,
        None,
    )
    .unwrap();
    let messages = request["messages"].as_array().unwrap();
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["tool_calls"].as_array().unwrap().len(), 2);
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call_f0a4339686c323854b75eb66");
    assert_eq!(messages[1]["tool_calls"][1]["id"], "call_85ee193b4a64700c420cadc4");
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "call_f0a4339686c323854b75eb66");
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["tool_call_id"], "call_85ee193b4a64700c420cadc4");
    assert_eq!(messages[4]["role"], "user");
    assert_eq!(messages[4]["content"][0]["type"], "image");
    assert_eq!(messages.len(), 5);
    let sha256 = messages[4]["content"][0]["sha256"].as_str().unwrap();
    assert_eq!(request["images"][sha256]["data"], TINY_PNG);
}

#[test]
fn unmatched_tool_result_without_an_assistant_turn_is_rejected() {
    let error = api::prepare(
        &json!({
            "model": "qwasar-qwen38-27b",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "tool", "tool_call_id": "call_missing", "content": "data"}
            ]
        }),
        false,
        None,
    )
    .unwrap_err();
    assert_eq!(error, "tool result does not match a pending call");
}

#[test]
fn codex_assistant_message_with_null_tool_calls_is_accepted() {
    let request = api::prepare(
        &json!({
            "model":"qwasar-qwen38-27b",
            "input":[
                {"type":"input_text","text":"keep going"},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will inspect the folder."}],
                    "tool_calls":null},
                {"type":"function_call","call_id":"call_exec","name":"exec_command",
                    "arguments":"{\"cmd\":\"ls\"}"},
                {"type":"function_call_output","call_id":"call_exec","output":"ok"}
            ]
        }),
        true,
        None,
    )
    .unwrap();
    assert_eq!(request["messages"][1]["role"], "assistant");
    assert!(request["messages"][1].get("tool_calls").is_none());
    assert_eq!(request["messages"][1]["content"], "I will inspect the folder.");
    assert_eq!(request["messages"][2]["tool_calls"][0]["id"], "call_exec");
    assert_eq!(request["messages"][3]["role"], "tool");
}

#[test]
fn store_keeps_images_by_hash() {
    let path = std::env::temp_dir().join(format!("qwasar-images-{}-{}.db", std::process::id(), api::new_id("test")));
    let store = Store::open(&path).unwrap();
    assert!(store.image("ff").unwrap().is_none());
    store.put_image("ff", "image/png", &[1, 2, 3]).unwrap();
    store.put_image("ff", "image/jpeg", &[9]).unwrap();
    assert_eq!(store.image("ff").unwrap().unwrap(), ("image/png".to_string(), vec![1, 2, 3]));
    drop(store);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn anthropic_messages_map_to_chat_and_echo_claude_aliases() {
    let request = anthropic::prepare(&json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 2048,
        "system": [{"type":"text","text":"coding","cache_control":{"type":"ephemeral"}}],
        "messages": [
            {"role":"user","content":"read src/main.rs"},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"look at the file","signature":"abc"},
                {"type":"text","text":"I'll read it."},
                {"type":"tool_use","id":"toolu_1","name":"Read","input":{"path":"src/main.rs"}}
            ]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"toolu_1","content":"fn main() {}"},
                {"type":"text","text":"now edit it"}
            ]}
        ],
        "tools": [{"name":"Read","description":"Read a file","input_schema":{
            "type":"object","properties":{"path":{"type":"string"}},"required":["path"]
        }}],
        "tool_choice": {"type":"auto"},
        "thinking": {"type":"enabled","budget_tokens":4096}
    }))
    .unwrap();
    assert_eq!(request["thinking"], "xhigh");
    assert_eq!(request["reasoning_budget_tokens"], 4096);
    assert_eq!(request["max_tokens"], 2048);
    assert_eq!(request["messages"][0]["content"], "coding");
    assert_eq!(request["messages"][1]["content"], "read src/main.rs");
    assert_eq!(request["messages"][2]["reasoning_content"], "look at the file");
    assert_eq!(request["messages"][2]["tool_calls"][0]["id"], "toolu_1");
    assert_eq!(request["messages"][2]["tool_calls"][0]["function"]["name"], "Read");
    assert_eq!(request["messages"][3]["role"], "tool");
    assert_eq!(request["messages"][3]["tool_call_id"], "toolu_1");
    assert_eq!(request["messages"][3]["content"], "fn main() {}");
    assert_eq!(request["messages"][4]["content"], "now edit it");
    assert_eq!(request["tools"][0]["function"]["name"], "Read");
    let disabled = anthropic::prepare(&json!({
        "model": api::MODEL,
        "messages": [{"role":"user","content":"hi"}],
        "thinking": {"type":"disabled"}
    }))
    .unwrap();
    assert_eq!(disabled["thinking"], "off");
    assert!(anthropic::prepare(&json!({
        "model":"gpt-5.5","messages":[{"role":"user","content":"hi"}]
    }))
    .is_err());
}

#[test]
fn anthropic_system_and_tool_roles_inside_messages_are_hoisted() {
    let request = anthropic::prepare(&json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "user", "content": "build tetris"},
            {"role": "system", "content": [
                {"type": "text", "text": "SessionStart hook context", "cache_control": {"type": "ephemeral"}},
                {"type": "tool_addition", "name": "Skill"}
            ]}
        ]
    }))
    .unwrap();
    assert_eq!(request["messages"][0]["role"], "system");
    assert_eq!(request["messages"][0]["content"], "SessionStart hook context");
    assert_eq!(request["messages"][1]["role"], "user");
    assert_eq!(request["messages"][1]["content"], "build tetris");
    let with_tool = anthropic::prepare(&json!({
        "model": api::MODEL,
        "max_tokens": 64,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_hook", "name": "Read", "input": {"path": "a"}}
            ]},
            {"role": "tool", "tool_use_id": "toolu_hook", "content": "ok"}
        ],
        "tools": [{"name": "Read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
    }))
    .unwrap();
    assert_eq!(with_tool["messages"][2]["role"], "tool");
    assert_eq!(with_tool["messages"][2]["tool_call_id"], "toolu_hook");
    assert_eq!(with_tool["messages"][2]["content"], "ok");
}

#[test]
fn anthropic_result_uses_content_blocks_and_count_tokens_avoids_the_worker() {
    let terminal = json!({
        "status":"completed",
        "message":{
            "role":"assistant",
            "content":"done",
            "reasoning_content":"plan",
            "tool_calls":[{"id":"toolu_2","type":"function","function":{"name":"Edit","arguments":"{\"path\":\"a\"}"}}]
        },
        "usage":{"prompt_tokens":10,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3}}
    });
    let message = anthropic::result("msg_1", "claude-sonnet-4-6", &terminal);
    assert_eq!(message["type"], "message");
    assert_eq!(message["model"], "claude-sonnet-4-6");
    assert_eq!(message["stop_reason"], "tool_use");
    assert_eq!(message["content"][0]["type"], "thinking");
    assert_eq!(message["content"][0]["thinking"], "plan");
    assert_eq!(message["content"][1]["text"], "done");
    assert_eq!(message["content"][2]["name"], "Edit");
    assert_eq!(message["content"][2]["input"]["path"], "a");
    assert_eq!(message["usage"]["input_tokens"], 10);
    assert_eq!(message["usage"]["cache_read_input_tokens"], 3);
    let tokens = anthropic::count_tokens(&json!({
        "model": api::MODEL,
        "messages": [{"role":"user","content":"hello world"}]
    }))
    .unwrap();
    assert!(tokens >= 1);
}

#[test]
fn anthropic_inline_images_and_tool_result_images_are_accepted() {
    let request = anthropic::prepare(&json!({
        "model": api::MODEL,
        "messages": [{"role":"user","content":[
            {"type":"text","text":"describe"},
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":TINY_PNG}}
        ]}]
    }))
    .unwrap();
    assert_eq!(request["messages"][0]["content"][0]["type"], "text");
    assert_eq!(request["messages"][0]["content"][1]["type"], "image");
    assert!(anthropic::prepare(&json!({
        "model": api::MODEL,
        "messages": [{"role":"user","content":[
            {"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}
        ]}]
    }))
    .is_err());
}
