use q38::artifact;
use q38::protocol::{Step, Worker};
use serde_json::json;
use std::env;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut model = None;
    let mut donor = None;
    let mut model_manifest = None;
    let mut donor_manifest = None;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let mut value = || args.next().unwrap_or_default();
        match argument.as_str() {
            "--model" => model = Some(PathBuf::from(value())),
            "--donor" => donor = Some(PathBuf::from(value())),
            "--model-manifest" => model_manifest = Some(PathBuf::from(value())),
            "--donor-manifest" => donor_manifest = Some(PathBuf::from(value())),
            "--help" | "-h" => {
                eprintln!("q38-worker --model PATH --donor PATH [--model-manifest PATH] [--donor-manifest PATH]");
                return ExitCode::from(2);
            }
            other => {
                eprintln!("q38-worker: unknown option {other}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(model), Some(donor)) = (model, donor) else {
        eprintln!("q38-worker: --model and --donor are required");
        return ExitCode::from(2);
    };
    let model_manifest = model_manifest.unwrap_or_else(|| find_manifest("benchmarks/manifests/qwen38-27b-rtx5090-v1.json"));
    let donor_manifest = donor_manifest.unwrap_or_else(|| find_manifest("benchmarks/manifests/nvidia-qwen38-27b-nvfp4.json"));
    let model_json = match std::fs::read_to_string(&model_manifest).and_then(|text| serde_json::from_str(&text).map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("q38-worker: cannot read {}: {error}", model_manifest.display());
            return ExitCode::from(1);
        }
    };
    let donor_json = match std::fs::read_to_string(&donor_manifest).and_then(|text| serde_json::from_str(&text).map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("q38-worker: cannot read {}: {error}", donor_manifest.display());
            return ExitCode::from(1);
        }
    };
    let loaded = match artifact::verify(&model, &donor, &model_json, &donor_json) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("q38-worker: {error}");
            return ExitCode::from(1);
        }
    };
    let worker = Worker::new();
    let mut config = worker.ready()["config"].clone();
    config["verified"] = json!(true);
    config["model_sha256"] = json!(loaded.model_sha256);
    config["donor_revision"] = json!(loaded.donor_revision);
    config["device_allocated"] = json!(false);
    config["reservation"] = json!({
        "kv_bytes": loaded.reservation.kv_bytes,
        "gdn_state_bytes": loaded.reservation.gdn_state_bytes,
        "vision_bytes": loaded.reservation.vision_bytes,
        "weight_budget_bytes": loaded.reservation.weight_budget_bytes,
        "sealed_bytes": loaded.reservation.sealed_bytes,
    });
    serve(&worker.ready_with(config))
}

fn serve(ready: &serde_json::Value) -> ExitCode {
    let mut worker = Worker::new();
    let mut output = std::io::stdout().lock();
    let Ok(line) = serde_json::to_string(ready) else { return ExitCode::from(1) };
    if writeln!(output, "{line}").is_err() || output.flush().is_err() {
        return ExitCode::from(1);
    }
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let (step, events) = worker.push(&line);
        for event in events {
            let Ok(encoded) = serde_json::to_string(&event) else { return ExitCode::from(1) };
            if writeln!(output, "{encoded}").is_err() {
                return ExitCode::from(1);
            }
        }
        if output.flush().is_err() || step == Step::Stop {
            break;
        }
    }
    ExitCode::SUCCESS
}

fn find_manifest(relative: &str) -> PathBuf {
    let mut candidates = Vec::new();
    if let Ok(current) = env::current_dir() {
        candidates.extend(current.ancestors().map(PathBuf::from));
    }
    if let Ok(executable) = env::current_exe() {
        candidates.extend(executable.ancestors().map(PathBuf::from));
    }
    candidates.into_iter().find(|root| root.join(relative).is_file()).map(|root| root.join(relative)).unwrap_or_else(|| PathBuf::from(relative))
}
