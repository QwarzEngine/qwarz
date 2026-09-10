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
fn ignored_controls_are_rejected_instead_of_silently_defaulted() {
    for additional in [
        json!({"parallel_tool_calls":false}),
        json!({"metadata":{"purpose":"must persist"}}),
        json!({"reasoning_effort":false}),
        json!({"reasoning":{"effort":"medium","summary":"detailed"}}),
        json!({"reasoning":"medium"}),
        json!({"reasoning_effort":"low","reasoning":{"effort":"high"}}),
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
