use crate::{
    api,
    store::Store,
    stream::ResponsesStream,
    worker::{Worker, WorkerConfig},
};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::Stream;
use serde_json::{Value, json};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::{
    sync::{mpsc, oneshot},
    time::Instant,
};

#[derive(Clone)]
pub struct App {
    pub worker: Arc<Worker>,
    pub store: Arc<Store>,
}

pub fn router(app: App) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/config", get(config))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/responses", post(responses))
        .route("/v1/responses/{id}", get(retrieve))
        .route("/v1/responses/{id}/cancel", post(cancel))
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(app)
}

async fn health(State(app): State<App>) -> Json<Value> {
    Json(json!({"service":"qwasar","version":"0.1.0","worker":app.worker.health()}))
}
async fn config(State(app): State<App>) -> Json<Value> {
    Json(app.worker.health())
}
async fn models() -> Json<Value> {
    Json(api::models_payload())
}
fn error(status: u16, code: &str, message: &str) -> Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(json!({"error":{"code":code,"message":message,"type":"invalid_request_error"}})),
    )
        .into_response()
}
fn terminal_error(terminal: &Value) -> Response {
    error(
        terminal["error"]["http_status"].as_u64().unwrap_or(503) as u16,
        terminal["error"]["code"]
            .as_str()
            .unwrap_or("generation_failed"),
        terminal["error"]["message"]
            .as_str()
            .unwrap_or("generation cancelled or failed"),
    )
}
fn failure(id: &str, code: &str, message: &str, status: &str) -> Value {
    json!({"type":"terminal","id":id,"status":status,"error":{"code":code,"message":message,"http_status":503}})
}
async fn retrieve(State(app): State<App>, Path(id): Path<String>) -> Response {
    match app.store.get(&id) {
        Ok(result) => Json(result.get("response").cloned().unwrap_or(result)).into_response(),
        Err(detail) => error(404, "not_found", &detail),
    }
}
async fn cancel(State(app): State<App>, Path(id): Path<String>) -> Response {
    match app.worker.cancel(&id).await {
        Ok(()) => Json(json!({"id":id,"status":"cancelling"})).into_response(),
        Err(detail) => error(409, "not_active", &detail),
    }
}
async fn chat(State(app): State<App>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    generate(app, headers, body, false).await
}
async fn responses(
    State(app): State<App>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    generate(app, headers, body, true).await
}

struct ClientStream {
    receiver: mpsc::Receiver<Result<Bytes, std::io::Error>>,
}
impl Stream for ClientStream {
    type Item = Result<Bytes, std::io::Error>;
    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(context)
    }
}
fn sse(value: &Value, responses: bool) -> Bytes {
    if responses {
        Bytes::from(format!(
            "event: {}\ndata: {value}\n\n",
            value["type"].as_str().unwrap_or("message")
        ))
    } else {
        Bytes::from(format!("data: {value}\n\n"))
    }
}
async fn send_frame(sender: &mpsc::Sender<Result<Bytes, std::io::Error>>, frame: Bytes) -> bool {
    matches!(
        tokio::time::timeout(Duration::from_secs(5), sender.send(Ok(frame))).await,
        Ok(Ok(()))
    )
}

async fn generate(app: App, headers: HeaderMap, body: Value, responses: bool) -> Response {
    let explicit_parent = match body.get("previous_response_id") {
        Some(parent) => match parent.as_str() {
            Some(id) => match app.store.parent(id) {
                Ok(snapshot) => Some(snapshot),
                Err(detail) => return error(400, "invalid_parent", &detail),
            },
            None => return error(400, "invalid_parent", "previous_response_id must be string"),
        },
        None => None,
    };
    let mut request = match api::prepare(&body, responses, explicit_parent.as_ref()) {
        Ok(request) => request,
        Err(detail) => return error(400, "invalid_request", &detail),
    };
    if let Err(response) = resolve_images(&app, &mut request) {
        return response;
    }
    let body = if request.get("images").is_some() {
        api::redact_images(&body)
    } else {
        body
    };
    let idempotency = headers
        .get("idempotency-key")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    if idempotency
        .as_ref()
        .is_some_and(|key| key.is_empty() || key.len() > 256)
    {
        return error(
            400,
            "invalid_idempotency_key",
            "key must contain 1..256 bytes",
        );
    }
    if let Some(key) = &idempotency {
        match app.store.idempotent(key) {
            Ok(Some((_id, previous, result))) => {
                if previous != body {
                    return error(
                        409,
                        "idempotency_conflict",
                        "key already belongs to another request",
                    );
                }
                if result["status"] == "in_progress" {
                    return error(409, "busy", "original request still running");
                }
                return replay_result(
                    result,
                    responses,
                    body["stream"] == true,
                    body["stream_options"]["include_usage"] != false,
                );
            }
            Err(detail) => return error(500, "storage_error", &detail),
            _ => {}
        }
    }
    let parent = if explicit_parent.is_some() {
        explicit_parent
    } else {
        match app
            .store
            .match_parent(request["messages"].as_array().unwrap())
        {
            Ok(parent) => parent,
            Err(detail) if detail.starts_with("ambiguous history") => {
                return error(400, "ambiguous_history", &detail);
            }
            Err(detail) => return error(500, "storage_error", &detail),
        }
    };
    let id = api::new_id("resp");
    let mut events = match app.worker.start(&id, request, parent).await {
        Ok(receiver) => receiver,
        Err(detail) => {
            return error(
                if detail == "busy" { 409 } else { 503 },
                "worker_unavailable",
                &detail,
            );
        }
    };
    if let Err(detail) = app.store.begin(&id, &body, idempotency.as_deref()) {
        tokio::spawn(async move {
            app.worker.reset(&id).await;
            app.worker.release(&id);
            drop(events);
        });
        return error(500, "storage_error", &detail);
    }
    let streaming = body["stream"] == true;
    let include_usage = body["stream_options"]["include_usage"] != false;
    let (output_sender, output_receiver) = mpsc::channel(128);
    let (ready_sender, ready_receiver) = oneshot::channel::<Result<(), Value>>();
    let (mut final_sender, final_receiver) = oneshot::channel::<Value>();
    let response_id = id.clone();
    tokio::spawn(async move {
        let mut ready_sender = Some(ready_sender);
        let mut response_stream = ResponsesStream::new(&id);
        let mut deadline = Instant::now() + app.worker.request_timeout;
        let mut cancellation_deadline = None;
        let mut forced_failure = None;
        let mut terminal;
        loop {
            if forced_failure.is_none() && app.worker.is_cancelled(&id) {
                forced_failure = Some(failure(
                    &id,
                    "cancelled",
                    "generation cancelled",
                    "cancelled",
                ));
                cancellation_deadline = Some(Instant::now() + app.worker.cancel_timeout);
            }
            let disconnected = async {
                if streaming {
                    output_sender.closed().await;
                } else {
                    final_sender.closed().await;
                }
            };
            let event = tokio::select! {
                biased;
                _ = disconnected, if forced_failure.is_none() => {
                    forced_failure = Some(failure(&id,"client_disconnected","client disconnected","cancelled"));
                    cancellation_deadline = Some(Instant::now() + app.worker.cancel_timeout);
                    let _ = app.worker.cancel(&id).await;
                    continue;
                }
                _ = tokio::time::sleep_until(cancellation_deadline.unwrap_or(deadline)) => {
                    if cancellation_deadline.is_some() {
                        app.worker.reset(&id).await;
                        terminal = forced_failure.take().unwrap();
                        break;
                    }
                    forced_failure = Some(failure(&id,"worker_timeout","worker event deadline exceeded","failed"));
                    cancellation_deadline = Some(Instant::now() + app.worker.cancel_timeout);
                    let _ = app.worker.cancel(&id).await;
                    continue;
                }
                event = events.recv() => event,
            };
            let Some(event) = event else {
                app.worker.reset(&id).await;
                terminal = forced_failure.take().unwrap_or_else(|| {
                    failure(&id, "worker_lost", "worker event stream closed", "failed")
                });
                break;
            };
            if event["type"] == "terminal" {
                terminal = forced_failure.take().unwrap_or(event);
                break;
            }
            deadline = Instant::now() + app.worker.request_timeout;
            if forced_failure.is_some() {
                continue;
            }
            let mut frames = Vec::new();
            if event["type"] == "started" {
                if let Some(sender) = ready_sender.take() {
                    let _ = sender.send(Ok(()));
                }
                if streaming {
                    frames.push(if responses {
                        response_stream.created()
                    } else {
                        api::chat_chunk(&id, json!({"role":"assistant"}), Value::Null)
                    });
                }
            }
            if event["type"] == "delta" && streaming {
                if responses {
                    frames = response_stream.delta(event["channel"] == "reasoning", &event["text"]);
                } else {
                    let mut delta = json!({});
                    delta[if event["channel"] == "reasoning" {
                        "reasoning_content"
                    } else {
                        "content"
                    }] = event["text"].clone();
                    frames.push(api::chat_chunk(&id, delta, Value::Null));
                }
            }
            if frames
                .iter()
                .any(|value| output_sender.try_send(Ok(sse(value, responses))).is_err())
            {
                forced_failure = Some(failure(
                    &id,
                    "client_backpressure",
                    "client could not keep up with the bounded stream",
                    "failed",
                ));
                cancellation_deadline = Some(Instant::now() + app.worker.cancel_timeout);
                let _ = app.worker.cancel(&id).await;
            }
        }
        terminal["created_at"] = json!(api::now());
        let mut response = response_stream.ordered_response(api::response_result(&id, &terminal));
        let status = terminal["status"].as_str().unwrap_or("failed");
        let stored = json!({"id":id,"status":status,"response":response,"chat":api::chat_result(&id,&terminal)});
        if let Err(detail) = app.store.finish(
            &id,
            status,
            &stored,
            terminal.get("snapshot").filter(|value| !value.is_null()),
        ) {
            terminal = json!({"status":"failed","error":{"code":"storage_error","message":detail,"http_status":500}});
            response = api::response_result(&id, &terminal);
            let failed = json!({"id":id,"status":"failed","response":response,"chat":api::chat_result(&id,&terminal)});
            let _ = app.store.finish(&id, "failed", &failed, None);
        }
        app.worker.release(&id);
        if let Some(sender) = ready_sender.take() {
            let _ = sender.send(Err(terminal.clone()));
        }
        if streaming {
            let mut frames = Vec::new();
            if responses {
                frames.extend(
                    response_stream
                        .finish(&response)
                        .iter()
                        .map(|event| sse(event, true)),
                );
            } else if matches!(terminal["status"].as_str(), Some("failed" | "cancelled")) {
                frames.push(sse(&json!({"error":terminal.get("error").filter(|value|!value.is_null()).cloned().unwrap_or(json!({"code":"cancelled","message":"generation cancelled"})),"id":id}),false));
            } else {
                if let Some(calls) = terminal["message"]["tool_calls"].as_array() {
                    for (index, call) in calls.iter().enumerate() {
                        let mut call = call.clone();
                        call["index"] = json!(index);
                        frames.push(sse(
                            &api::chat_chunk(&id, json!({"tool_calls":[call]}), Value::Null),
                            false,
                        ));
                    }
                }
                frames.push(sse(
                    &api::chat_chunk(&id, json!({}), json!(api::finish_reason(&terminal))),
                    false,
                ));
                if include_usage {
                    frames.push(sse(&json!({"id":id,"object":"chat.completion.chunk","created":api::now(),"model":api::MODEL,"choices":[],"usage":terminal["usage"],"qwasar_metrics":terminal["metrics"]}),false));
                }
            }
            if !responses {
                frames.push(Bytes::from_static(b"data: [DONE]\n\n"));
            }
            for frame in frames {
                if !send_frame(&output_sender, frame).await {
                    break;
                }
            }
        }
        let _ = final_sender.send(terminal);
    });
    match ready_receiver.await {
        Ok(Ok(())) => {}
        Ok(Err(terminal)) => return terminal_error(&terminal),
        Err(_) => return error(503, "worker_lost", "generation task ended unexpectedly"),
    }
    if streaming {
        return Response::builder()
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("x-accel-buffering", "no")
            .header("x-request-id", response_id)
            .body(Body::from_stream(ClientStream {
                receiver: output_receiver,
            }))
            .unwrap();
    }
    match final_receiver.await {
        Ok(terminal)
            if matches!(
                terminal["status"].as_str(),
                Some("completed" | "incomplete")
            ) =>
        {
            Json(if responses {
                api::response_result(&response_id, &terminal)
            } else {
                api::chat_result(&response_id, &terminal)
            })
            .into_response()
        }
        Ok(terminal) => terminal_error(&terminal),
        Err(_) => error(503, "worker_lost", "generation task ended"),
    }
}

/// Persists inline image bytes and fills hash-only references from the store so
/// the worker request is self-contained. Unknown hashes are rejected before
/// any generation is dispatched.
fn resolve_images(app: &App, request: &mut Value) -> Result<(), Response> {
    use base64::Engine;
    let Some(images) = request.get_mut("images").and_then(Value::as_object_mut) else {
        return Ok(());
    };
    let standard = base64::engine::general_purpose::STANDARD;
    for (sha256, image) in images.iter_mut() {
        let media_type = image["media_type"].as_str().unwrap_or("").to_string();
        match image["data"].as_str() {
            Some(data) => {
                let bytes = standard
                    .decode(data)
                    .map_err(|_| error(400, "invalid_request", "image data is not valid base64"))?;
                app.store
                    .put_image(sha256, &media_type, &bytes)
                    .map_err(|detail| error(500, "storage_error", &detail))?;
            }
            None => {
                let (stored_type, bytes) = app
                    .store
                    .image(sha256)
                    .map_err(|detail| error(500, "storage_error", &detail))?
                    .ok_or_else(|| {
                        error(400, "unknown_image", &format!("image {sha256} is not stored; resend it inline"))
                    })?;
                image["media_type"] = json!(stored_type);
                image["data"] = json!(standard.encode(bytes));
            }
        }
    }
    Ok(())
}

fn replay_result(stored: Value, responses: bool, streaming: bool, include_usage: bool) -> Response {
    if matches!(stored["status"].as_str(), Some("failed" | "cancelled")) {
        return terminal_error(stored.get("response").unwrap_or(&stored));
    }
    let result = stored[if responses { "response" } else { "chat" }].clone();
    if result.is_null() {
        return error(
            409,
            "incomplete_record",
            "stored request has no terminal payload",
        );
    }
    if !streaming {
        return Json(result).into_response();
    }
    let id = stored["id"].as_str().unwrap_or("unknown");
    let mut data = Vec::new();
    if responses {
        let mut stream = ResponsesStream::new(id);
        data.extend_from_slice(&sse(&stream.created(), true));
        for event in stream.finish(&result) {
            data.extend_from_slice(&sse(&event, true));
        }
    } else {
        let mut message = result["choices"][0]["message"].clone();
        if let Some(calls) = message["tool_calls"].as_array_mut() {
            for (index, call) in calls.iter_mut().enumerate() {
                call["index"] = json!(index);
            }
        }
        data.extend_from_slice(&sse(&api::chat_chunk(id, message, Value::Null), false));
        data.extend_from_slice(&sse(
            &api::chat_chunk(id, json!({}), result["choices"][0]["finish_reason"].clone()),
            false,
        ));
        if include_usage {
            data.extend_from_slice(&sse(&json!({"id":id,"object":"chat.completion.chunk","created":api::now(),"model":api::MODEL,"choices":[],"usage":result["usage"]}),false));
        }
        data.extend_from_slice(b"data: [DONE]\n\n");
    }
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from(data))
        .unwrap()
}

pub async fn serve(
    address: std::net::SocketAddr,
    database: &std::path::Path,
    config: WorkerConfig,
) -> Result<(), String> {
    if !address.ip().is_loopback() {
        return Err("v1 only binds loopback; remote unauthenticated access is prohibited".into());
    }
    if let Some(parent) = database
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|error| error.to_string())?;
    let ownership = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(database)
        .map_err(|error| error.to_string())?;
    ownership
        .try_lock()
        .map_err(|error| format!("database is owned by another server: {error}"))?;
    let store = Arc::new(Store::open(database)?);
    let worker = Worker::spawn(config);
    let app = App {
        worker: worker.clone(),
        store,
    };
    println!(
        "{}",
        json!({"service":"qwasar","address":listener.local_addr().unwrap().to_string()})
    );
    let result = axum::serve(listener, router(app))
        .with_graceful_shutdown(async move {
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
            tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
            worker.shutdown().await;
        })
        .await
        .map_err(|error| error.to_string());
    drop(ownership);
    result
}
