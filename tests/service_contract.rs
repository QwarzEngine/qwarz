use qwasar_server::{api, store::Store};
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
        json!({"response_format":{"type":"json_object"}}),
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
    let mut incomplete = terminal.clone();
    incomplete["status"] = json!("incomplete");
    assert_eq!(
        api::chat_result("resp_b", &incomplete)["choices"][0]["finish_reason"],
        "length"
    );
    let response = api::response_result("resp_a", &terminal);
    assert_eq!(response["output"][1]["type"], "function_call");
    assert_eq!(response["output"][1]["call_id"], "call_a");
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
    let many: Vec<_> = (0..=api::MAX_IMAGES).map(|_| image.clone()).collect();
    assert!(api::prepare(&json!({"model":"qwasar-qwen38-27b","messages":[{"role":"user","content":many}]}), false, None).is_err());
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
