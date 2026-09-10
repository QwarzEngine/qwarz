use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use qwasar_server::{
    api,
    service::{self, App},
    store::Store,
    worker::{Worker, WorkerConfig},
};
use serde_json::{Value, json};
use std::{
    os::unix::fs::PermissionsExt,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
use tower::ServiceExt;

fn spawn_worker(mode: &str) -> Arc<Worker> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/protocol_worker.py");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    Worker::spawn(WorkerConfig {
        python: path.to_str().unwrap().into(),
        model: mode.into(),
        prefill: "baseline".into(),
        context_size: 1024,
        fake: true,
        request_timeout: Duration::from_secs(600),
        cancel_timeout: Duration::from_secs(30),
    })
}

async fn ready(worker: &Worker) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while worker.health()["status"] != "ready" {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

async fn idle(worker: &Worker) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while worker.health()["busy"] == true {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn idle_worker_keeps_same_process_after_fifteen_minutes() {
    let worker = spawn_worker("ordinary");
    ready(&worker).await;
    let original_pid = worker.pid.load(Ordering::Acquire);
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(901)).await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    tokio::time::resume();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let healthy =
        worker.health()["status"] == "ready" && worker.pid.load(Ordering::Acquire) == original_pid;
    worker.shutdown().await;
    assert!(
        healthy,
        "idle loaded worker was killed by the protocol read timeout"
    );
}

#[tokio::test]
async fn dropping_start_during_blocked_stdin_recovers_admission() {
    let worker = spawn_worker("input-stall");
    ready(&worker).await;
    let dispatch_worker = worker.clone();
    let task = tokio::spawn(async move {
        dispatch_worker
            .start(
                "abandoned",
                json!({"messages":[{"role":"user","content":"x".repeat(2 * 1024 * 1024)}]}),
                None,
            )
            .await
    });
    tokio::time::sleep(Duration::from_millis(80)).await;
    task.abort();
    let _ = task.await;
    let recovered = tokio::time::timeout(Duration::from_secs(2), idle(&worker))
        .await
        .is_ok();
    worker.shutdown().await;
    assert!(
        recovered,
        "abandoned dispatch permanently retained busy admission"
    );
}

#[tokio::test]
async fn overflowing_client_stream_never_reports_truncated_success() {
    let worker = spawn_worker("burst");
    ready(&worker).await;
    let store = Arc::new(Store::open(std::path::Path::new(":memory:")).unwrap());
    let app = service::router(App {
        worker: worker.clone(),
        store,
    });
    let response = app.oneshot(Request::post("/v1/chat/completions").header("content-type", "application/json")
        .body(Body::from(json!({"model":api::MODEL,"messages":[{"role":"user","content":"hello"}],"stream":true}).to_string())).unwrap()).await.unwrap();
    assert_eq!(response.status(), 200);
    idle(&worker).await;
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let wire = String::from_utf8(bytes.to_vec()).unwrap();
    let mut text = String::new();
    let mut successful = false;
    let mut failed = false;
    for line in wire.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if let Ok(event) = serde_json::from_str::<Value>(line) {
            text.push_str(
                event["choices"][0]["delta"]["content"]
                    .as_str()
                    .unwrap_or(""),
            );
            successful |= event["choices"][0]["finish_reason"] == "stop";
            failed |= event.get("error").is_some();
        }
    }
    worker.shutdown().await;
    assert!(
        failed || (successful && text.len() == 512),
        "stream falsely completed with {} of 512 characters",
        text.len()
    );
}

#[tokio::test]
async fn request_timeout_reaps_unresponsive_child_before_releasing_busy() {
    let worker = spawn_worker("ignore-cancel");
    ready(&worker).await;
    let original_pid = worker.pid.load(Ordering::Acquire);
    let store = Arc::new(Store::open(std::path::Path::new(":memory:")).unwrap());
    let app = service::router(App {
        worker: worker.clone(),
        store: store.clone(),
    });
    let response = app.clone().oneshot(Request::post("/v1/chat/completions").header("content-type", "application/json")
        .body(Body::from(json!({"model":api::MODEL,"messages":[{"role":"user","content":"stall"}],"stream":true}).to_string())).unwrap()).await.unwrap();
    let id = response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .to_string();
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(601)).await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    let premature_release =
        worker.health()["busy"] == false && worker.pid.load(Ordering::Acquire) == original_pid;
    tokio::time::advance(Duration::from_secs(31)).await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    tokio::time::resume();
    idle(&worker).await;
    ready(&worker).await;
    assert_ne!(worker.pid.load(Ordering::Acquire), original_pid);
    assert!(!std::path::Path::new(&format!("/proc/{original_pid}")).exists());
    assert!(store.parent(&id).is_err());
    let wire = response.into_body().collect().await.unwrap().to_bytes();
    assert!(String::from_utf8_lossy(&wire).contains("worker_timeout"));
    let next = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":api::MODEL,"messages":[{"role":"user","content":"hello"}]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(next.status(), 200);
    worker.shutdown().await;
    assert!(
        !premature_release,
        "timeout released admission while original GPU child still lived"
    );
}

#[tokio::test]
async fn disconnected_nonstreaming_prefill_cancels_without_waiting_for_a_delta() {
    let worker = spawn_worker("prefill-stall");
    ready(&worker).await;
    let original_pid = worker.pid.load(Ordering::Acquire);
    let store = Arc::new(Store::open(std::path::Path::new(":memory:")).unwrap());
    let app = service::router(App {
        worker: worker.clone(),
        store,
    });
    let task = tokio::spawn(
        app.oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":api::MODEL,"messages":[{"role":"user","content":"stall"}]})
                        .to_string(),
                ))
                .unwrap(),
        ),
    );
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(worker.health()["busy"], true);
    task.abort();
    let _ = task.await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(31)).await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    tokio::time::resume();
    idle(&worker).await;
    assert_ne!(worker.pid.load(Ordering::Acquire), original_pid);
    worker.shutdown().await;
}
