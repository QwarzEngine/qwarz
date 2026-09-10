use serde_json::{Value, json};
use std::{
    process::Stdio,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{ChildStdin, Command},
    sync::{Notify, mpsc, watch},
};

#[derive(Clone)]
pub struct WorkerConfig {
    pub python: String,
    pub model: String,
    pub prefill: String,
    pub context_size: usize,
    pub fake: bool,
    pub request_timeout: Duration,
    pub cancel_timeout: Duration,
}

pub struct Worker {
    writer: tokio::sync::Mutex<Option<ChildStdin>>,
    pending: Mutex<Option<(String, mpsc::Sender<Value>)>>,
    cancelled: Mutex<Option<String>>,
    status: RwLock<Value>,
    busy: AtomicBool,
    stopping: AtomicBool,
    resetting: AtomicBool,
    changed: Notify,
    restart: Notify,
    reaped: watch::Sender<u64>,
    pub pid: AtomicU32,
    pub request_timeout: Duration,
    pub cancel_timeout: Duration,
}

impl Worker {
    pub fn spawn(config: WorkerConfig) -> Arc<Self> {
        let worker = Arc::new(Self {
            writer: tokio::sync::Mutex::new(None),
            pending: Mutex::new(None),
            cancelled: Mutex::new(None),
            status: RwLock::new(json!({"status":"starting"})),
            busy: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            resetting: AtomicBool::new(false),
            changed: Notify::new(),
            restart: Notify::new(),
            reaped: watch::channel(0).0,
            pid: AtomicU32::new(0),
            request_timeout: config.request_timeout,
            cancel_timeout: config.cancel_timeout,
        });
        tokio::spawn(worker.clone().run(config));
        worker
    }

    pub fn health(&self) -> Value {
        let mut status = self.status.read().unwrap().clone();
        status["busy"] = json!(self.busy.load(Ordering::Acquire));
        status["pid"] = json!(self.pid.load(Ordering::Acquire));
        status
    }

    async fn write(&self, command: Value) -> Result<(), String> {
        tokio::time::timeout(Duration::from_secs(10), async {
            let mut writer = self.writer.lock().await;
            let writer = writer.as_mut().ok_or("worker unavailable".to_string())?;
            let data = format!("{command}\n");
            writer
                .write_all(data.as_bytes())
                .await
                .map_err(|error| error.to_string())?;
            writer.flush().await.map_err(|error| error.to_string())
        })
        .await
        .map_err(|_| "worker input timeout".to_string())?
    }

    pub async fn start(
        self: &Arc<Self>,
        id: &str,
        request: Value,
        parent: Option<Value>,
    ) -> Result<mpsc::Receiver<Value>, String> {
        if self.health()["status"] != "ready" || self.stopping.load(Ordering::Acquire) {
            return Err("worker unavailable".into());
        }
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err("busy".into());
        }
        let (sender, receiver) = mpsc::channel(256);
        *self.pending.lock().unwrap() = Some((id.to_string(), sender.clone()));
        *self.cancelled.lock().unwrap() = None;
        self.changed.notify_one();
        let worker = self.clone();
        let id = id.to_string();
        tokio::spawn(async move {
            let failed = tokio::select! {
                biased;
                _ = sender.closed() => true,
                result = worker.write(json!({"op":"generate","id":id,"request":request,"parent":parent})) => result.is_err(),
            };
            if failed {
                worker.reset(&id).await;
                if sender.is_closed() {
                    worker.release(&id);
                }
            }
        });
        Ok(receiver)
    }

    pub fn is_cancelled(&self, id: &str) -> bool {
        self.cancelled.lock().unwrap().as_deref() == Some(id)
    }

    pub async fn cancel(&self, id: &str) -> Result<(), String> {
        let pending = self.pending.lock().unwrap().clone();
        let Some((active, sender)) = pending.filter(|(active, _)| active == id) else {
            return Err("response is not active".into());
        };
        *self.cancelled.lock().unwrap() = Some(active);
        let _ = sender.try_send(json!({"type":"cancelling","id":id}));
        self.write(json!({"op":"cancel","id":id})).await
    }

    pub async fn reset(&self, id: &str) {
        let mut reaped = self.reaped.subscribe();
        if !self
            .pending
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|(active, _)| active == id)
        {
            return;
        }
        *self.status.write().unwrap() = json!({"status":"restarting"});
        self.resetting.store(true, Ordering::Release);
        self.restart.notify_one();
        let _ = reaped.changed().await;
    }

    pub fn release(&self, id: &str) {
        let mut pending = self.pending.lock().unwrap();
        if pending.as_ref().is_some_and(|(active, _)| active == id) {
            *pending = None;
            self.busy.store(false, Ordering::Release);
            self.changed.notify_one();
        }
    }

    pub async fn shutdown(&self) {
        self.stopping.store(true, Ordering::Release);
        self.restart.notify_one();
    }

    async fn run(self: Arc<Self>, config: WorkerConfig) {
        while !self.stopping.load(Ordering::Acquire) {
            *self.status.write().unwrap() = json!({"status":"loading"});
            let mut command = Command::new(&config.python);
            command.args([
                "-u",
                "-m",
                "qwasar_runtime.worker",
                "--model",
                &config.model,
                "--prefill",
                &config.prefill,
                "--context-size",
                &config.context_size.to_string(),
            ]);
            if config.fake {
                command.arg("--fake");
            }
            command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .kill_on_drop(true);
            match command.spawn() {
                Err(error) => {
                    *self.status.write().unwrap() =
                        json!({"status":"failed","error":error.to_string()});
                }
                Ok(mut child) => {
                    self.pid.store(child.id().unwrap_or(0), Ordering::Release);
                    *self.writer.lock().await = child.stdin.take();
                    let mut reader = BufReader::new(child.stdout.take().unwrap()).lines();
                    loop {
                        if self.stopping.load(Ordering::Acquire)
                            || self.resetting.load(Ordering::Acquire)
                        {
                            break;
                        }
                        let pending = self.pending.lock().unwrap().clone();
                        let disconnected = async {
                            match &pending {
                                Some((_, sender)) => sender.closed().await,
                                None => std::future::pending().await,
                            }
                        };
                        let read = tokio::select! {
                            biased;
                            _ = self.restart.notified() => continue,
                            _ = self.changed.notified() => continue,
                            _ = disconnected => break,
                            read = reader.next_line() => read,
                        };
                        let Ok(Some(line)) = read else {
                            break;
                        };
                        if line.len() > 16 * 1024 * 1024 {
                            break;
                        }
                        let Ok(event) = serde_json::from_str::<Value>(&line) else {
                            eprintln!("worker emitted invalid JSON protocol");
                            break;
                        };
                        if event["type"] == "ready" {
                            if event["protocol"] != 1 {
                                break;
                            }
                            *self.status.write().unwrap() =
                                json!({"status":"ready","config":event["config"]});
                            continue;
                        }
                        let pending = self.pending.lock().unwrap().clone();
                        if let Some((id, sender)) = pending {
                            if event["id"] != id {
                                eprintln!("worker response ID mismatch");
                                break;
                            }
                            let sent = tokio::select! {
                                _ = self.restart.notified() => false,
                                sent = tokio::time::timeout(Duration::from_secs(5), sender.send(event)) => matches!(sent, Ok(Ok(()))),
                            };
                            if !sent {
                                break;
                            }
                        }
                    }
                    *self.status.write().unwrap() = json!({"status":"restarting"});
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    *self.writer.lock().await = None;
                    self.pid.store(0, Ordering::Release);
                    self.resetting.store(false, Ordering::Release);
                    self.reaped.send_modify(|epoch| *epoch += 1);
                    let pending = self.pending.lock().unwrap().clone();
                    if let Some((id, sender)) = pending {
                        if sender.is_closed() {
                            self.release(&id);
                        } else {
                            let _ = tokio::time::timeout(Duration::from_secs(5), sender.send(json!({"type":"terminal","id":id,"status":"failed","error":{"code":"worker_lost","message":"GPU worker exited; reconstruct from last committed response after restart","http_status":503}}))).await;
                        }
                    }
                }
            }
            if !self.stopping.load(Ordering::Acquire) {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {},
                    _ = self.restart.notified() => {
                        self.resetting.store(false, Ordering::Release);
                        self.reaped.send_modify(|epoch| *epoch += 1);
                    },
                }
            }
        }
        *self.status.write().unwrap() = json!({"status":"stopped"});
    }
}
