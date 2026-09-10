use qwasar_server::{service, worker::WorkerConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("qwasar: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let mut config = WorkerConfig {
        python: std::env::var("QWASAR_EXLLAMA_PYTHON")
            .unwrap_or("../qwen38-exl3-mia/.venv/bin/python".into()),
        model: std::env::var("QWASAR_MODEL_PATH")
            .unwrap_or("/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw".into()),
        prefill: "flash".into(),
        context_size: 262144,
        fake: false,
        request_timeout: std::time::Duration::from_secs(600),
        cancel_timeout: std::time::Duration::from_secs(30),
    };
    let mut host = "127.0.0.1".to_string();
    let mut port = 8800_u16;
    let mut database = PathBuf::from("state/qwasar.db");
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--fake-worker" {
            config.fake = true;
            continue;
        }
        if argument == "--help" {
            println!(
                "qwasar-server [--host 127.0.0.1] [--port 8800] [--database state/qwasar.db] [--python PATH] [--model PATH] [--prefill flash|baseline] [--context-size 262144] [--request-timeout-secs 600] [--cancel-timeout-secs 30] [--fake-worker]"
            );
            return Ok(());
        }
        let value = arguments
            .next()
            .ok_or(format!("missing value for {argument}"))?;
        match argument.as_str() {
            "--host" => host = value,
            "--port" => port = value.parse().map_err(|_| "invalid port")?,
            "--database" => database = value.into(),
            "--python" => config.python = value,
            "--model" => config.model = value,
            "--prefill" => config.prefill = value,
            "--context-size" => {
                config.context_size = value.parse().map_err(|_| "invalid context size")?
            }
            "--request-timeout-secs" | "--cancel-timeout-secs" => {
                let seconds: u64 = value.parse().map_err(|_| "invalid timeout")?;
                if seconds == 0 || seconds > 86400 {
                    return Err("timeout must be 1..86400 seconds".into());
                }
                if argument == "--request-timeout-secs" {
                    config.request_timeout = std::time::Duration::from_secs(seconds);
                } else {
                    config.cancel_timeout = std::time::Duration::from_secs(seconds);
                }
            }
            _ => return Err(format!("unknown option {argument}")),
        }
    }
    if !["flash", "baseline"].contains(&config.prefill.as_str())
        || config.context_size == 0
        || config.context_size > 262144
        || config.context_size % 256 != 0
    {
        return Err("invalid profile or context size".into());
    }
    let address = format!("{host}:{port}")
        .parse()
        .map_err(|_| "invalid listen address")?;
    service::serve(address, &database, config).await
}
