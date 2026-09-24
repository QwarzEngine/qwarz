mod setup;

use setup::Options;

const DELEGATED: [&str; 5] = ["serve", "stop", "status", "logs", "wait-ready"];

fn usage() {
    println!(
        r#"qwarz — Qwen3.8-27B on one RTX 5090

usage: qwarz <command> [options]

commands:
  start               install and run the service: detect the GPU (RTX 5090
                      only), verify or download the pinned artifacts, build,
                      install the systemd unit and wait until the worker is
                      ready; also installs the `qwarz` command on PATH
  explain             show the engine decisions — every lever, its measurement
                      and its rollback switch — plus the current boot state
  status              service status
  logs                follow the service logs
  stop                stop the service
  serve, wait-ready   internal commands used by the systemd unit

start options:
  --gpu N             select a specific GPU index (must be an RTX 5090);
                      default: the first RTX 5090 found
  --model PATH        EXL3 artifact directory
                      (default: $QWASAR_MODEL_PATH or ~/models/Qwen3.8-27B-EXL3-5.0bpw)
  --python PATH       ExLlamaV3 venv python
                      (default: $QWASAR_EXLLAMA_PYTHON or
                       ~/Documents/llm/qwen38-exl3-mia/.venv/bin/python)
  --donor PATH        NVIDIA64 NVFP4 donor directory
                      (default: $QWASAR_NVIDIA_DONOR or the path pinned in
                       benchmarks/manifests/nvidia-qwen38-27b-nvfp4.json)
  --download          download missing artifacts from Hugging Face
  --skip-hashes       skip SHA-256 verification of large artifacts
  --yes, -y           do not prompt for confirmation
  --help              show this help"#
    );
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let Some(command) = arguments.next() else {
        usage();
        std::process::exit(2);
    };
    match command.as_str() {
        "--help" | "-h" => usage(),
        "start" => run_start(&mut arguments),
        "explain" => {
            if let Err(error) = setup::explain() {
                eprintln!("qwarz: {error}");
                std::process::exit(1);
            }
        }
        other if DELEGATED.contains(&other) => setup::delegate(other, arguments),
        other => {
            eprintln!("qwarz: unknown command {other}");
            usage();
            std::process::exit(2);
        }
    }
}

fn run_start(arguments: &mut impl Iterator<Item = String>) {
    let mut options = Options::default();
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .unwrap_or_else(|| exit_usage(&format!("missing value for {argument}")))
        };
        match argument.as_str() {
            "--download" => options.download = true,
            "--skip-hashes" => options.skip_hashes = true,
            "--yes" | "-y" => options.yes = true,
            "--gpu" => match value().parse::<u32>() {
                Ok(index) => options.gpu = Some(index),
                Err(_) => exit_usage("--gpu must be a non-negative integer"),
            },
            "--model" => options.model = Some(std::path::PathBuf::from(value())),
            "--python" => options.python = Some(std::path::PathBuf::from(value())),
            "--donor" => options.donor = Some(std::path::PathBuf::from(value())),
            other => exit_usage(&format!("unknown option {other} for start")),
        }
    }
    if let Err(error) = setup::start(&options) {
        eprintln!("qwarz: {error}");
        std::process::exit(1);
    }
}

fn exit_usage(message: &str) -> ! {
    eprintln!("qwarz: {message}");
    usage();
    std::process::exit(2);
}
