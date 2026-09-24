//! One-command installer for the RTX 5090 production stack.
//!
//! `qwasar-setup start` walks six phases, each failing fast with a precise
//! message: hardware detection (RTX 5090 only), environment validation, the
//! pinned artifact checks (with optional Hugging Face downloads), the release
//! build, the systemd user unit, and the readiness wait. Nothing here touches
//! the inference stack itself: every lever it enables is exactly the promoted
//! recipe already shipped in `integrations/systemd/qwasar.service`.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

pub const EXL3_REPOSITORY: &str = "thelastspark/Qwen3.8-27B-exl3";
pub const EXL3_REVISION: &str = "1a6fe4afb5b921fda9f93fd4b06d6c6d5c99a62c";

const MODEL_MANIFEST: &str = "benchmarks/manifests/qwen38-27b-rtx5090-v1.json";
const DONOR_MANIFEST: &str = "benchmarks/manifests/nvidia-qwen38-27b-nvfp4.json";
const DEFAULT_MODEL: &str = "models/Qwen3.8-27B-EXL3-5.0bpw";
const DEFAULT_DONOR: &str = "models/nvidia-qwen38-27b-nvfp4";
const DEFAULT_PYTHON: &str = "Documents/llm/qwen38-exl3-mia/.venv/bin/python";
const DEFAULT_REPO: &str = "Documents/llm/qwarz";
const FLASHINFER_PATHS: &[&str] = &[
    "results/20260908-upstream-experiments/fp8/pscaled",
    "results/20260908-upstream-experiments/fp8/deps",
    "results/20260908-upstream-experiments/fp8/deps/nvidia_cutlass_dsl/dsl_packages",
    "results/20260908-hybrid-backends/flashinfer-deps",
    "results/20260908-hybrid-backends/flashinfer-deps/nvidia_cutlass_dsl/dsl_packages",
];
const UNIT_NAME: &str = "qwasar.service";
const HOST: &str = "127.0.0.1";
const PORT: u16 = 8800;
const READY_TIMEOUT_SECS: u64 = 540;
const FAILURE_GRACE_SECS: u64 = 120;
const OCCUPIED_MIB: u64 = 512;
const MIN_5090_MIB: u64 = 30000;
const CONTEXT_SIZE: u64 = 262144;

#[derive(Default)]
pub struct Options {
    pub gpu: Option<u32>,
    pub model: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub donor: Option<PathBuf>,
    pub download: bool,
    pub skip_hashes: bool,
    pub yes: bool,
}

#[derive(Debug)]
pub struct SetupError(pub String);

impl std::fmt::Display for SetupError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl From<std::io::Error> for SetupError {
    fn from(error: std::io::Error) -> Self {
        SetupError(error.to_string())
    }
}

struct Gpu {
    index: u32,
    uuid: String,
    name: String,
    memory_mib: u64,
    driver: String,
}

struct Resolved {
    repo: PathBuf,
    home: PathBuf,
    gpu: Gpu,
    model: PathBuf,
    python: PathBuf,
    donor: PathBuf,
    donor_custom: bool,
    cuda_home: Option<PathBuf>,
}

pub fn start(options: &Options) -> Result<(), SetupError> {
    let repo = discover_repo()?;
    let home = home_directory()?;
    println!("==> [1/6] Hardware");
    let gpu = detect_gpu(options)?;
    let resolved = resolve(&repo, &home, options, gpu)?;
    println!("==> [2/6] Environment");
    validate_environment(&resolved)?;
    println!("==> [3/6] Model artifacts");
    ensure_artifacts(&resolved, options)?;
    println!("==> [4/6] Build");
    build_server(&resolved.repo)?;
    println!("==> [5/6] Systemd service");
    install_systemd(&resolved, options)?;
    println!("==> [6/6] Waiting for the worker");
    wait_ready()?;
    print_summary(&resolved);
    Ok(())
}

const EXPLAIN: &str = r#"Qwarz — Qwen3.8-27B on one RTX 5090, native 262,144-token context, one
persistent agentic coding session. Every lever below was measured against a
same-day control before it shipped; docs/benchmarks/ holds the campaign
record and results/ the data.

Model and quantization
  EXL3 5bpw artifact (thelastspark/Qwen3.8-27B-exl3 @ 1a6fe4a), SHA-256
  verified at every load. Attention (Q/K/V/O, GDN), embeddings, lm_head, the
  MTP head and the vision tower (BF16) are the quality-sensitive parts, so
  they stay on the finer EXL3 quants (5 bpw, head 6-bit, MTP 4-bit).
  NVIDIA64 NVFP4 donor (nvidia/Qwen3.8-27B-NVFP4 @ dbb8f4) supplies
  gate/up/down of all 64 layers (192 matrices) as Blackwell-native FP4.
  There is no merged hybrid artifact: the graft is per-module at load time,
  and the engine refuses to serve unless the donor shapes match exactly, all
  192 modules were replaced and 64 per-layer MLP CUDA graphs were captured.
  Why this split: MLPs carry ~two thirds of the forward matmul FLOPs and
  NVFP4 is the fastest native format on the RTX 5090, while the
  fine-grained parts keep quality. Artifact and donor hashes are baked into
  the session identity: swapping either invalidates every session.

Speculative decoding
  MTP6: six fixed draft proposals per cycle, ~0.62–0.70 acceptance on coding.
  Rendezvous: the speculative loop stays resident on the GPU. Draft ids
  never round-trip to the host (one sync per cycle instead of ~12) and the
  verify window is sampled in one batched pass. Cost: +2.5 GB VRAM.
  64K proposer head (hot64k-20260922): the drafter reads a 65536-token head
  (512 complete Hadamard groups, 6 bpw, frequency-calibrated map pinned by
  SHA-256); the verifier keeps the full 248320-row head, so a missing id
  merely stops being proposable. Constant −2.5 ms/verify of draft weight
  reads: decode +11–17% at 4K–128K, TTFT par, quality gate 4/4. +0.24 GB
  VRAM; if the install fails at boot the service degrades to the full head
  with a different session identity (reported in /health as mtp_head).
  Draft-loop CUDA graph (draft-graph-20260923): the six-step draft walk is
  captured as one CUDA graph replay per verify. −0.9..−1.1 ms/verify at 32K,
  +5.6% tok/s at 256K, memory delta 0. With the GDN determinism patch the
  stack is bit-reproducible; greedy verification makes completions identical
  even where graph-vs-eager draft ids diverge at near-tie states.

Attention and prefill
  XQA decode attention with per-layer decode CUDA graphs, on the NVFP4
  one-level KV cache (page 256) — deep-context reads ride the FP4 path.
  PRIMS FP8 prefill for large prefills (≥8K): 5,300–6,800 tok/s.
  The whole promoted stack vs the pre-XQA flash stack: decode +35% @32K,
  +47% @256K, TTFT −3–4%.

Operational decisions
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — the 262K stack rides
  within ~100 MiB of device-full and the native allocator's fragmentation
  turned that into transient OOMs on long prompts.
  HF_HUB_OFFLINE=1 — artifacts are local and hash-pinned; serving never
  touches the network.
  One GPU worker, batch 1: concurrent requests get 409. The Rust/SQLite
  supervisor owns sessions, exact token history, cancellation and recovery;
  a persistent ExLlamaV3 worker owns the GPU.

Rejected or capped along the way (the gates did the rejecting)
  DFlash2 drafter A/B: cycle cost beat its acceptance gain.
  XQA-seam extension: stayed capped below its promotion gate.

Rollback and kill switches (all at boot)
  --prefill flash in the unit rolls back to the pre-XQA stack.
  QWASAR_RDZ=0 disables the rendezvous path; QWASAR_HOT64K=0 the 64K
  proposer head; QWASAR_DRAFT_GRAPH=0 the draft CUDA graph."#;

pub fn explain() -> Result<(), SetupError> {
    println!("{EXPLAIN}");
    match health() {
        Some(response) => {
            let config = &response["worker"]["config"];
            let installed = |value: &Value| value.as_bool() == Some(true);
            println!("Current boot (live from /health):");
            println!("  context:          {} tokens", config["context_size"]);
            println!("  quantization:     {}", text(&config["quantization"]));
            println!("  prefill:          {} (PRIMS >= {} tok)", text(&config["prefill"]), config["prims_min_query"]);
            println!("  decode attention: {}", text(&config["decode_attention"]));
            println!("  KV cache:         {}", text(&config["target_cache"]));
            println!("  MLP CUDA graphs:  {}/64", config["mlp_graphs"]);
            if installed(&config["mtp_head"]["installed"]) {
                let head = &config["mtp_head"];
                println!(
                    "  MTP:              {} draft tokens, 64K proposer head ({}/{} groups exact, map {})",
                    config["draft_tokens"],
                    head["weight_mapping"]["groups_exact"],
                    head["weight_mapping"]["groups"],
                    text(&head["map_sha256"])
                );
            } else {
                println!("  MTP:              {} draft tokens, full 248320-token proposer head (degraded)", config["draft_tokens"]);
            }
            if installed(&config["rendezvous"]["installed"]) {
                println!("  rendezvous:       installed (GPU-resident draft chain)");
            } else {
                println!("  rendezvous:       off");
            }
            if installed(&config["draft_graph"]["installed"]) {
                println!("  draft graph:      {}", text(&config["draft_graph"]["revision"]));
            } else {
                println!("  draft graph:      off");
            }
        }
        None => println!("The service is not running; run `qwarz start` and ask again for the live boot state."),
    }
    Ok(())
}

pub fn delegate(command: &str, arguments: impl Iterator<Item = String>) -> ! {
    let script = match discover_repo() {
        Ok(repo) => repo.join("scripts/qwasar.py"),
        Err(error) => {
            eprintln!("qwarz: {error}");
            std::process::exit(1);
        }
    };
    let status = Command::new("python3").arg(&script).arg(command).args(arguments).status();
    match status {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(error) => {
            eprintln!("qwarz: cannot run {}: {error}", script.display());
            std::process::exit(1);
        }
    }
}

fn home_directory() -> Result<PathBuf, SetupError> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| SetupError("HOME is not set".into()))
}

fn discover_repo() -> Result<PathBuf, SetupError> {
    let mut candidates = Vec::new();
    if let Ok(executable) = std::env::current_exe() {
        candidates.extend(executable.ancestors().map(Path::to_path_buf));
    }
    if let Ok(directory) = std::env::current_dir() {
        candidates.extend(directory.ancestors().map(Path::to_path_buf));
    }
    candidates
        .into_iter()
        .find(|root| {
            root.join("Cargo.toml").is_file() && root.join(DONOR_MANIFEST).is_file()
        })
        .ok_or_else(|| SetupError("cannot locate the qwarz repository root; run qwasar-setup from a checkout".into()))
}

fn resolve(repo: &Path, home: &Path, options: &Options, gpu: Gpu) -> Result<Resolved, SetupError> {
    let model = options
        .model
        .clone()
        .or_else(|| environment_path("QWASAR_MODEL_PATH"))
        .unwrap_or_else(|| home.join(DEFAULT_MODEL));
    let python = options
        .python
        .clone()
        .or_else(|| environment_path("QWASAR_EXLLAMA_PYTHON"))
        .unwrap_or_else(|| home.join(DEFAULT_PYTHON));
    let (donor, donor_custom) = match options.donor.clone().or_else(|| environment_path("QWASAR_NVIDIA_DONOR")) {
        Some(path) => (path, true),
        None => {
            let pin = donor_pin(repo)?;
            let manifest_path = PathBuf::from(pin["path"].as_str().unwrap_or_default());
            if manifest_path.is_dir() {
                (manifest_path, false)
            } else {
                let revision = pin["revision"].as_str().unwrap_or_default().to_string();
                (home.join(DEFAULT_DONOR).join(revision), true)
            }
        }
    };
    Ok(Resolved {
        repo: repo.to_path_buf(),
        home: home.to_path_buf(),
        gpu,
        model,
        python,
        donor,
        donor_custom,
        cuda_home: locate_cuda(),
    })
}

fn environment_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn detect_gpu(options: &Options) -> Result<Gpu, SetupError> {
    let output = capture(
        "nvidia-smi",
        &[
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
    )?;
    let mut gpus = Vec::new();
    for line in output.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 5 {
            continue;
        }
        let (index, memory) = match (fields[0].parse::<u32>(), fields[3].trim_end_matches(" MiB").parse::<u64>()) {
            (Ok(index), Ok(memory)) => (index, memory),
            _ => continue,
        };
        gpus.push(Gpu {
            index,
            uuid: fields[1].to_string(),
            name: fields[2].to_string(),
            memory_mib: memory,
            driver: fields[4].to_string(),
        });
    }
    if gpus.is_empty() {
        return Err(SetupError("nvidia-smi reported no GPUs".into()));
    }
    for gpu in &gpus {
        let marker = if gpu.name.contains("RTX 5090") && gpu.memory_mib >= MIN_5090_MIB {
            "eligible"
        } else if gpu.name.contains("RTX 5090") {
            "RTX 5090 with insufficient VRAM"
        } else {
            "not an RTX 5090"
        };
        println!(
            "    GPU {}: {} ({} MiB, driver {}) [{}]",
            gpu.index, gpu.name, gpu.memory_mib, gpu.driver, marker
        );
    }
    let chosen = match options.gpu {
        Some(index) => gpus
            .iter()
            .find(|gpu| gpu.index == index)
            .filter(|gpu| gpu.name.contains("RTX 5090") && gpu.memory_mib >= MIN_5090_MIB)
            .ok_or_else(|| SetupError(format!("GPU {index} is not an eligible RTX 5090; Qwasar v1 requires one RTX 5090 (32 GB)")))?,
        None => gpus
            .iter()
            .find(|gpu| gpu.name.contains("RTX 5090") && gpu.memory_mib >= MIN_5090_MIB)
            .ok_or_else(|| SetupError("no RTX 5090 found; Qwasar v1 requires one RTX 5090 (32 GB)".into()))?,
    };
    let others = gpus
        .iter()
        .filter(|gpu| gpu.index != chosen.index && gpu.name.contains("RTX 5090") && gpu.memory_mib >= MIN_5090_MIB)
        .count();
    if others > 0 {
        println!(
            "    multiple RTX 5090s found; using GPU {} (pass --gpu N to choose another)",
            chosen.index
        );
    }
    let apps = capture(
        "nvidia-smi",
        &[
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    .unwrap_or_default();
    let mut service_occupier = false;
    for line in apps.lines() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 3 || fields[0] != chosen.uuid {
            continue;
        }
        if let Ok(used) = fields[2].parse::<u64>() {
            if used >= OCCUPIED_MIB {
                if is_qwasar_process(fields[1]) {
                    service_occupier = true;
                    continue;
                }
                return Err(SetupError(format!(
                    "GPU {} is occupied by PID {} ({} MiB); stop its owner explicitly first",
                    chosen.index, fields[1], used
                )));
            }
        }
    }
    if service_occupier {
        println!("    GPU is held by the running qwasar service; start will restart it");
    }
    println!("    selected GPU {}: {}", chosen.index, chosen.name);
    Ok(Gpu {
        index: chosen.index,
        uuid: chosen.uuid.clone(),
        name: chosen.name.clone(),
        memory_mib: chosen.memory_mib,
        driver: chosen.driver.clone(),
    })
}

fn is_qwasar_process(pid: &str) -> bool {
    std::fs::read(format!("/proc/{pid}/cmdline"))
        .map(|bytes| bytes.windows(6).any(|window| window == b"qwasar"))
        .unwrap_or(false)
}

fn locate_cuda() -> Option<PathBuf> {
    if let Some(cuda_home) = environment_path("CUDA_HOME") {
        if cuda_home.join("bin/nvcc").is_file() {
            return Some(cuda_home);
        }
    }
    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        if directory.join("nvcc").is_file() {
            return directory.parent().map(Path::to_path_buf);
        }
    }
    let fallback = PathBuf::from("/opt/cuda");
    (fallback.join("bin/nvcc").is_file()).then_some(fallback)
}

fn validate_environment(resolved: &Resolved) -> Result<(), SetupError> {
    if !resolved.python.is_file() {
        return Err(SetupError(format!(
            "ExLlamaV3 venv python not found at {}; set --python or QWASAR_EXLLAMA_PYTHON (the runtime venv normally comes from the sibling qwen38-exl3-mia checkout)",
            resolved.python.display()
        )));
    }
    let version = capture(&resolved.python.to_string_lossy(), &["--version"])?;
    let parsed = version
        .trim()
        .strip_prefix("Python ")
        .and_then(|text| text.split('.').next().and_then(|major| major.parse::<u32>().ok()))
        .and_then(|major| version.split('.').nth(1).and_then(|minor| minor.trim().parse::<u32>().ok()).map(|minor| (major, minor)));
    let (major, minor) = parsed.ok_or_else(|| SetupError(format!("cannot parse python version: {version:?}")))?;
    if major != 3 || minor < 12 {
        return Err(SetupError(format!("python {major}.{minor} found; the runtime requires >= 3.12")));
    }
    println!("    Python {major}.{minor} at {}", resolved.python.display());
    capture(&resolved.python.to_string_lossy(), &["-c", "import exllamav3"]).map_err(|_| {
        SetupError(format!(
            "exllamav3 is not importable in {}; rebuild the sibling qwen38-exl3-mia venv before installing",
            resolved.python.display()
        ))
    })?;
    println!("    exllamav3 imports cleanly");
    capture("cargo", &["--version"])
        .map_err(|_| SetupError("cargo is not in PATH; install the Rust toolchain".into()))?;
    let cuda_home = resolved.cuda_home.clone().ok_or_else(|| {
        SetupError("CUDA toolkit not found (nvcc); the PRIMS/XQA prefill stack needs nvcc".into())
    })?;
    println!("    CUDA toolkit at {}", cuda_home.display());
    let mut missing = Vec::new();
    for relative in FLASHINFER_PATHS {
        let path = resolved.repo.join(relative);
        if !path.is_dir() {
            missing.push(path);
        }
    }
    if !missing.is_empty() {
        let listed = missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(SetupError(format!(
            "FlashInfer hybrid trees missing: {listed}; they ship with the lever campaigns (see docs/benchmarks/2026-09-20-serving-evaluation.md)"
        )));
    }
    println!("    FlashInfer hybrid trees: {}/{} present", FLASHINFER_PATHS.len(), FLASHINFER_PATHS.len());
    Ok(())
}

fn donor_pin(repo: &Path) -> Result<Value, SetupError> {
    let manifest = fs::read_to_string(repo.join(DONOR_MANIFEST))?;
    serde_json::from_str(&manifest).map_err(|error| SetupError(format!("donor manifest is not valid JSON: {error}")))
}

fn model_pin(repo: &Path) -> Result<Value, SetupError> {
    let manifest = fs::read_to_string(repo.join(MODEL_MANIFEST))?;
    serde_json::from_str(&manifest).map_err(|error| SetupError(format!("model manifest is not valid JSON: {error}")))
}

fn hf_cli(python: &Path) -> Option<PathBuf> {
    let bin = python.parent()?;
    ["hf", "huggingface-cli"]
        .iter()
        .map(|name| bin.join(name))
        .find(|candidate| candidate.is_file())
}

fn hf_download(cli: &Path, repository: &str, revision: &str, directory: &Path) -> Result<(), SetupError> {
    fs::create_dir_all(directory)?;
    println!("    downloading {repository} @ {revision} into {}", directory.display());
    let status = Command::new(cli)
        .args(["download", repository, "--revision", revision])
        .arg("--local-dir")
        .arg(directory)
        .status()?;
    if !status.success() {
        return Err(SetupError(format!("download of {repository} failed (exit {status}); gated repositories may need a Hugging Face token")));
    }
    Ok(())
}

fn has_safetensors(directory: &Path) -> bool {
    fs::read_dir(directory)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|entry| entry.file_name().to_string_lossy().ends_with(".safetensors"))
        })
        .unwrap_or(false)
}

fn ensure_artifacts(resolved: &Resolved, options: &Options) -> Result<(), SetupError> {
    let pin = model_pin(&resolved.repo)?;
    let expected_sha256 = pin["model"]["artifact_sha256"]
        .as_str()
        .ok_or_else(|| SetupError("model manifest is missing model.artifact_sha256".into()))?
        .to_string();
    let mut downloaded = false;
    if !resolved.model.is_dir() || !resolved.model.join("config.json").is_file() || !has_safetensors(&resolved.model) {
        if !options.download {
            return Err(SetupError(format!(
                "EXL3 artifact missing at {}; re-run with --download to fetch {EXL3_REPOSITORY} @ {EXL3_REVISION}",
                resolved.model.display()
            )));
        }
        let cli = hf_cli(&resolved.python).ok_or_else(|| {
            SetupError("no huggingface-cli in the ExLlamaV3 venv; install huggingface_hub there or download manually".into())
        })?;
        hf_download(&cli, EXL3_REPOSITORY, EXL3_REVISION, &resolved.model)?;
        downloaded = true;
    }
    let artifact_sha256 = verify_model(&resolved.model, options.skip_hashes, downloaded, &expected_sha256)?;
    println!("    EXL3 artifact sha256 {artifact_sha256}");
    let donor = donor_pin(&resolved.repo)?;
    let revision = donor["revision"]
        .as_str()
        .ok_or_else(|| SetupError("donor manifest is missing revision".into()))?
        .to_string();
    let repository = donor["repository"]
        .as_str()
        .ok_or_else(|| SetupError("donor manifest is missing repository".into()))?
        .to_string();
    if !resolved.donor.is_dir() {
        if !options.download {
            return Err(SetupError(format!(
                "NVIDIA64 donor missing at {}; re-run with --download to fetch {repository} @ {revision}",
                resolved.donor.display()
            )));
        }
        let cli = hf_cli(&resolved.python).ok_or_else(|| {
            SetupError("no huggingface-cli in the ExLlamaV3 venv; install huggingface_hub there or download manually".into())
        })?;
        hf_download(&cli, &repository, &revision, &resolved.donor)?;
    }
    verify_donor(&resolved.donor, &donor, options.skip_hashes)?;
    println!("    NVFP4 donor revision {revision} ({} shards checked)", donor["shards"].as_object().map_or(0, serde_json::Map::len));
    Ok(())
}

fn verify_model(model: &Path, skip_hashes: bool, downloaded: bool, expected: &str) -> Result<String, SetupError> {
    let config_text = fs::read_to_string(model.join("config.json"))
        .map_err(|error| SetupError(format!("cannot read {}: {error}", model.join("config.json").display())))?;
    let config: Value = serde_json::from_str(&config_text)
        .map_err(|error| SetupError(format!("model config.json is not valid JSON: {error}")))?;
    let quantization = &config["quantization_config"];
    if quantization["quant_method"].as_str() != Some("exl3") || quantization["bits"].as_f64() != Some(5.0) {
        return Err(SetupError(format!(
            "artifact at {} is not the pinned EXL3 5bpw quantization; expected {EXL3_REPOSITORY} @ {EXL3_REVISION}",
            model.display()
        )));
    }
    let native = config["text_config"]["max_position_embeddings"]
        .as_u64()
        .or_else(|| config["max_position_embeddings"].as_u64())
        .unwrap_or(0);
    if native < CONTEXT_SIZE {
        return Err(SetupError(format!(
            "artifact native context {native} is below the required {CONTEXT_SIZE}"
        )));
    }
    if skip_hashes && !downloaded {
        return Ok("skipped (--skip-hashes)".into());
    }
    let digest = sha256_path(model)?;
    if digest != expected {
        return Err(SetupError(format!(
            "artifact hash mismatch at {}: expected {expected}, got {digest}",
            model.display()
        )));
    }
    Ok(digest)
}

fn verify_donor(donor: &Path, pin: &Value, skip_hashes: bool) -> Result<(), SetupError> {
    let shards = pin["shards"]
        .as_object()
        .ok_or_else(|| SetupError("donor manifest is missing shards".into()))?;
    for (name, info) in shards {
        let shard = donor.join(name);
        if !shard.is_file() {
            return Err(SetupError(format!("donor shard missing: {}", shard.display())));
        }
        let expected_size = info["size"].as_u64().unwrap_or_default();
        let size = fs::metadata(&shard)?.len();
        if size != expected_size {
            return Err(SetupError(format!(
                "donor shard size mismatch: {name}: {size} != {expected_size}"
            )));
        }
        if skip_hashes {
            continue;
        }
        println!("    hashing {name}");
        let digest = sha256_file(&shard)?;
        let expected = info["sha256"].as_str().unwrap_or_default();
        if digest != expected {
            return Err(SetupError(format!(
                "donor shard hash mismatch: {name}: expected {expected}, got {digest}"
            )));
        }
    }
    Ok(())
}

fn build_server(repo: &Path) -> Result<(), SetupError> {
    println!("    cargo build --release --locked (this can take a while the first time)");
    let status = Command::new("cargo")
        .args(["build", "--release", "--locked"])
        .current_dir(repo)
        .status()?;
    if !status.success() {
        return Err(SetupError(format!("cargo build failed (exit {status})")));
    }
    let binary = repo.join("target/release/qwasar-server");
    if !binary.is_file() {
        return Err(SetupError(format!("{} was not produced by the build", binary.display())));
    }
    Ok(())
}

fn unit_text(resolved: &Resolved) -> String {
    let cuda = resolved
        .cuda_home
        .as_deref()
        .unwrap_or_else(|| Path::new("/opt/cuda"))
        .display();
    let donor_line = if resolved.donor_custom {
        format!("Environment=\"QWASAR_NVIDIA_DONOR={}\"", resolved.donor.display())
    } else {
        String::new()
    };
    let spacer = if donor_line.is_empty() { "" } else { "\n" };
    format!(
        r#"[Unit]
Description=Qwasar Qwen3.8-27B EXL3 + NVIDIA64 + XQA/PRIMS + MTP6 + NVFP4-KV (RTX 5090)
StartLimitIntervalSec=0

[Service]
Type=exec
WorkingDirectory={repo}
Environment="PATH={cuda}/bin:/usr/local/bin:/usr/bin"
Environment="CUDA_VISIBLE_DEVICES={gpu}"
Environment="CUDA_HOME={cuda}"
Environment="QWASAR_MODEL_PATH={model}"
Environment="QWASAR_EXLLAMA_PYTHON={python}"{spacer}{donor_line}
Environment="HF_HUB_OFFLINE=1"
Environment="PYTHONUNBUFFERED=1"
# The 262K-context stack rides within ~100 MiB of device-full; expandable
# segments grow in place instead of requiring fresh contiguous blocks.
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
ExecStart=/usr/bin/python3 {repo}/scripts/qwasar.py serve --prefill xqa
ExecStartPost=/usr/bin/python3 {repo}/scripts/qwasar.py wait-ready --pid ${{MAINPID}}
Restart=on-failure
RestartSec=15
TimeoutStartSec=600
TimeoutStopSec=45
KillMode=mixed
UMask=0077
StandardOutput=journal
StandardError=journal
SyslogIdentifier=qwasar

[Install]
WantedBy=default.target
"#,
        repo = resolved.repo.display(),
        gpu = resolved.gpu.index,
        model = resolved.model.display(),
        python = resolved.python.display(),
    )
}

fn matches_unit_defaults(resolved: &Resolved) -> bool {
    resolved.repo == resolved.home.join(DEFAULT_REPO)
        && resolved.gpu.index == 0
        && resolved.model == resolved.home.join(DEFAULT_MODEL)
        && resolved.python == resolved.home.join(DEFAULT_PYTHON)
        && !resolved.donor_custom
        && resolved.cuda_home.as_deref() == Some(Path::new("/opt/cuda"))
}

fn install_systemd(resolved: &Resolved, options: &Options) -> Result<(), SetupError> {
    let unit_directory = resolved.home.join(".config/systemd/user");
    fs::create_dir_all(&unit_directory)?;
    let unit = unit_directory.join(UNIT_NAME);
    let repo_unit = resolved.repo.join("integrations/systemd").join(UNIT_NAME);
    let defaults = matches_unit_defaults(resolved);
    let replace = if unit.symlink_metadata().is_ok() {
        let points_to_repo_unit = fs::canonicalize(&unit).is_ok_and(|target| target == fs::canonicalize(&repo_unit).unwrap_or(repo_unit.clone()));
        if defaults && points_to_repo_unit {
            println!("    unit already links the repository template");
            false
        } else if options.yes || confirm(&format!("replace {} with a unit generated for this machine", unit.display())) {
            let backup = unit.with_extension("service.pre-setup.bak");
            let _ = fs::rename(&unit, &backup);
            println!("    previous unit backed up to {}", backup.display());
            true
        } else {
            return Err(SetupError("refusing to replace the existing unit".into()));
        }
    } else {
        true
    };
    if replace {
        if defaults {
            std::os::unix::fs::symlink(&repo_unit, &unit)?;
            println!("    unit installed as a symlink to {}", repo_unit.display());
            println!("    scripts/qwasar.py start/stop/status continue to manage it via systemctl");
        } else {
            fs::write(&unit, unit_text(resolved))?;
            println!("    generated {} with non-default paths", unit.display());
            println!("    note: manage this unit with systemctl; scripts/qwasar.py start/stop only delegate to systemctl when standard paths are used");
        }
    }
    install_command_link(resolved)?;
    capture("systemctl", &["--user", "daemon-reload"])?;
    let state = tolerant_output("systemctl", &["--user", "is-active", UNIT_NAME]);
    if state == "active" || state == "activating" || state == "reloading" {
        let already_ready =
            health().is_some_and(|response| response["worker"]["status"] == "ready");
        if already_ready && !replace {
            println!("    service already running and ready; leaving it up");
        } else {
            println!("    service is {state}; restarting it");
            if run_systemctl(&["restart", UNIT_NAME]).is_err() {
                // The stop boundary can race the old worker's GPU release and
                // the first start may bounce; the unit retries on its own and
                // wait_ready decides whether the service ever comes up.
                println!("    restart reported a transient failure; waiting for the unit's own retry");
            }
        }
    } else {
        if health().is_some() {
            return Err(SetupError(format!(
                "port {PORT} is already serving another process; stop it before installing"
            )));
        }
        run_systemctl(&["enable", "--now", UNIT_NAME])?;
    }
    Ok(())
}

fn install_command_link(resolved: &Resolved) -> Result<(), SetupError> {
    let binary = resolved.repo.join("target/release/qwarz");
    if !binary.is_file() {
        return Err(SetupError(format!("{} was not produced by the build", binary.display())));
    }
    let directory = resolved.home.join(".local/bin");
    fs::create_dir_all(&directory)?;
    let link = directory.join("qwarz");
    if fs::read_link(&link).is_ok_and(|target| target == binary) {
        return Ok(());
    }
    let _ = fs::remove_file(&link);
    std::os::unix::fs::symlink(&binary, &link)?;
    println!("    command installed as `qwarz` ({})", link.display());
    let path = std::env::var_os("PATH").unwrap_or_default();
    if !std::env::split_paths(&path).any(|entry| entry == directory) {
        println!("    note: {} is not in PATH yet; add it to use `qwarz` directly", directory.display());
    }
    Ok(())
}

fn run_systemctl(arguments: &[&str]) -> Result<(), SetupError> {
    let mut command = vec!["--user"];
    command.extend(arguments.iter().copied());
    let full: Vec<&str> = command;
    let status = Command::new("systemctl").args(&full).status()?;
    if !status.success() {
        return Err(SetupError(format!(
            "systemctl {} failed (exit {})",
            full[2..].join(" "),
            status.code().unwrap_or(-1)
        )));
    }
    Ok(())
}

fn tolerant_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_default()
}

fn confirm(question: &str) -> bool {
    print!("    {question}? [y/N] ");
    let _ = std::io::stdout().flush();
    let mut answer = String::new();
    let _ = std::io::stdin().read_line(&mut answer);
    matches!(answer.trim(), "y" | "Y" | "yes" | "Yes")
}

fn health() -> Option<Value> {
    let mut stream = TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], PORT)),
        Duration::from_millis(500),
    )
    .ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let request = format!("GET /health HTTP/1.0\r\nHost: {HOST}:{PORT}\r\n\r\n");
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).ok()?;
    let text = String::from_utf8_lossy(&response);
    let body = text.split_once("\r\n\r\n")?.1;
    serde_json::from_str(body.trim()).ok()
}

fn wait_ready() -> Result<(), SetupError> {
    let mut consecutive_failures = 0u64;
    for _ in 0..READY_TIMEOUT_SECS {
        if let Some(response) = health() {
            if response["worker"]["status"] == "ready" {
                println!("    worker ready on {HOST}:{PORT}/v1");
                return Ok(());
            }
        }
        // A stop/start boundary can race the old worker's GPU release; the
        // unit retries on its own (Restart=on-failure, StartLimitIntervalSec=0),
        // so only a sustained failed state is an error.
        if tolerant_output("systemctl", &["--user", "is-failed", UNIT_NAME]) == "failed" {
            consecutive_failures += 1;
            if consecutive_failures >= FAILURE_GRACE_SECS {
                return Err(SetupError(
                    "the service stays in a failed state; inspect journalctl --user -u qwasar.service -n 100".into(),
                ));
            }
        } else {
            consecutive_failures = 0;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    Err(SetupError(format!(
        "startup still pending after {READY_TIMEOUT_SECS}s; inspect journalctl --user -u qwasar.service -n 100"
    )))
}

fn print_summary(resolved: &Resolved) {
    let donor = donor_pin(&resolved.repo).ok();
    let revision = donor
        .as_ref()
        .and_then(|pin| pin["revision"].as_str())
        .unwrap_or("unknown");
    println!(
        r#"
Qwasar is running.

  GPU:      {} (GPU {}, {} MiB, driver {})
  Model:    {EXL3_REPOSITORY} @ {} (5.0 bpw) — {}
  Donor:    {} — {}
  Stack:    NVFP4 MLP + XQA decode + PRIMS prefill + MTP6 + 64K proposer head + draft CUDA graph
  Context:  262,144 tokens
  API:      http://{HOST}:{PORT}/v1 (model: qwasar-qwen38-27b)

  Manage:   qwarz status
  Logs:     qwarz logs
  Stop:     qwarz stop
  Why:      qwarz explain"#,
        resolved.gpu.name,
        resolved.gpu.index,
        resolved.gpu.memory_mib,
        resolved.gpu.driver,
        &EXL3_REVISION[..7],
        resolved.model.display(),
        revision,
        resolved.donor.display(),
    );
}

fn capture(program: &str, arguments: &[&str]) -> Result<String, SetupError> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| SetupError(format!("cannot run {program}: {error}")))?;
    if !output.status.success() {
        let message = String::from_utf8_lossy(&output.stderr);
        return Err(SetupError(format!(
            "{program} {} failed (exit {}): {}",
            arguments.join(" "),
            output.status,
            message.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn text(value: &Value) -> &str {
    value.as_str().unwrap_or("-")
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_file(path: &Path) -> Result<String, SetupError> {
    let file = fs::File::open(path)?;
    let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex(digest.finalize()))
}

/// Byte-for-byte replica of `qwasar_bench.environment.sha256_path` so the
/// installer verifies the same digest the engine checks at every load.
fn sha256_path(source: &Path) -> Result<String, SetupError> {
    let mut digest = Sha256::new();
    let root: PathBuf;
    let mut paths: Vec<PathBuf> = Vec::new();
    if source.is_file() {
        root = source.parent().unwrap_or(Path::new(".")).to_path_buf();
        paths.push(source.to_path_buf());
    } else if source.is_dir() {
        root = source.to_path_buf();
        collect_files(source, &root, &mut paths)?;
    } else {
        return Err(SetupError(format!("path not found: {}", source.display())));
    }
    paths.sort_by(|left, right| left.to_string_lossy().cmp(&right.to_string_lossy()));
    for item in paths {
        let relative = item
            .strip_prefix(&root)
            .map_err(|error| SetupError(error.to_string()))?
            .to_string_lossy();
        let bytes = relative.as_bytes();
        digest.update(&(bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
        let size = fs::metadata(&item)?.len();
        digest.update(&size.to_be_bytes());
        let file = fs::File::open(&item)?;
        let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);
        let mut buffer = [0u8; 1024 * 1024];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
    }
    Ok(hex(digest.finalize()))
}

fn collect_files(directory: &Path, root: &Path, out: &mut Vec<PathBuf>) -> Result<(), SetupError> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_files(&path, root, out)?;
        } else if entry.file_type()?.is_file() {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let excluded = relative
                .components()
                .any(|component| component.as_os_str() == ".cache" || component.as_os_str() == ".git");
            let name = relative.file_name().map(|name| name.to_string_lossy().to_string()).unwrap_or_default();
            let transient = name.ends_with(".part") || name.ends_with(".lock") || name.ends_with(".incomplete");
            if !excluded && !transient {
                out.push(path);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_path_matches_python_reference() {
        // Fixture and expected digest produced by
        // qwasar_bench.environment.sha256_path over the same tree under /tmp;
        // the digest embeds the root path, so it assumes /tmp as the tempdir.
        let root = std::env::temp_dir().join("qwasar-setup-sha256-reference");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join(".cache")).unwrap();
        fs::create_dir_all(root.join("nested")).unwrap();
        fs::write(root.join("config.json"), b"{\"bits\": 5.0}").unwrap();
        fs::write(root.join("nested/shard.safetensors"), b"weights").unwrap();
        fs::write(root.join("partial.part"), b"ignored").unwrap();
        fs::write(root.join(".cache/ignored.bin"), b"ignored").unwrap();
        let digest = sha256_path(&root).unwrap();
        assert_eq!(
            digest,
            "4d36b74af0089375358a6716f7b4a77c626bd52f9b65c651b91070a2ad1be078",
            "the digest must match the Python reference implementation"
        );
    }

    #[test]
    fn unit_text_pins_production_recipe() {
        let resolved = Resolved {
            repo: PathBuf::from("/repo"),
            home: PathBuf::from("/home"),
            gpu: Gpu { index: 1, uuid: "u".into(), name: "NVIDIA GeForce RTX 5090".into(), memory_mib: 32607, driver: "610".into() },
            model: PathBuf::from("/home/models/m"),
            python: PathBuf::from("/home/venv/bin/python"),
            donor: PathBuf::from("/home/models/d"),
            donor_custom: true,
            cuda_home: Some(PathBuf::from("/opt/cuda")),
        };
        let text = unit_text(&resolved);
        assert!(text.contains("CUDA_VISIBLE_DEVICES=1"));
        assert!(text.contains("QWASAR_NVIDIA_DONOR=/home/models/d"));
        assert!(text.contains("serve --prefill xqa"));
        assert!(text.contains("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"));
        assert!(text.contains("wait-ready --pid ${MAINPID}"));
    }
}
