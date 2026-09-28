//! Pinned EXL3 artifact and NVIDIA64 donor. Verification only: a mismatch
//! refuses to become ready, and this phase does not allocate device memory.

use crate::profile::{self, CONTEXT_TOKENS};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub struct Reservation {
    pub kv_bytes: u64,
    pub gdn_state_bytes: u64,
    pub vision_bytes: u64,
    pub weight_budget_bytes: u64,
    pub sealed_bytes: u64,
    pub device_allocated: bool,
}

pub struct Loaded {
    pub model_sha256: String,
    pub donor_revision: String,
    pub reservation: Reservation,
}

pub fn reservation() -> Reservation {
    Reservation {
        kv_bytes: profile::kv_bytes(CONTEXT_TOKENS),
        gdn_state_bytes: profile::GDN_STATE_BYTES,
        vision_bytes: profile::VISION_BUDGET_BYTES,
        weight_budget_bytes: profile::WEIGHT_BUDGET_BYTES,
        sealed_bytes: profile::sealed_bytes(CONTEXT_TOKENS),
        device_allocated: false,
    }
}

pub fn verify(model: &Path, donor: &Path, model_manifest: &Value, donor_manifest: &Value) -> Result<Loaded, String> {
    let expected = model_manifest["model"]["artifact_sha256"]
        .as_str()
        .ok_or("model manifest is missing model.artifact_sha256")?;
    let actual = sha256_path(model)?;
    if actual != expected {
        return Err(format!("artifact hash mismatch: expected {expected}, got {actual}"));
    }
    let config = fs::read_to_string(model.join("config.json")).map_err(|error| error.to_string())?;
    let config: Value = serde_json::from_str(&config).map_err(|error| error.to_string())?;
    let text = config.get("text_config").filter(|value| value.is_object()).unwrap_or(&config);
    let positions = text["max_position_embeddings"].as_u64().unwrap_or(0);
    if positions < CONTEXT_TOKENS {
        return Err(format!("artifact native context is {positions}, need {CONTEXT_TOKENS}"));
    }
    let quant = &config["quantization_config"];
    let bits = quant["bits"].as_f64();
    if quant["quant_method"] != "exl3" || bits != Some(5.0) {
        return Err("v1 requires the pinned EXL3 5bpw artifact".into());
    }
    let revision = donor_manifest["revision"].as_str().ok_or("donor manifest is missing revision")?;
    let shards = donor_manifest["shards"].as_object().ok_or("donor manifest is missing shards")?;
    if !donor.is_dir() {
        return Err(format!("NVIDIA donor missing: {}", donor.display()));
    }
    for (name, expected) in shards {
        let path = donor.join(name);
        let metadata = fs::metadata(&path).map_err(|_| format!("NVIDIA shard missing: {}", path.display()))?;
        let expected_size = expected["size"].as_u64().ok_or_else(|| format!("donor shard {name} has no size"))?;
        if metadata.len() != expected_size {
            return Err(format!("NVIDIA shard size mismatch: {name}: {} != {expected_size}", metadata.len()));
        }
        let expected_hash = expected["sha256"].as_str().ok_or_else(|| format!("donor shard {name} has no sha256"))?;
        let actual_hash = sha256_file(&path)?;
        if actual_hash != expected_hash {
            return Err(format!("NVIDIA shard hash mismatch: {name}"));
        }
    }
    let plan = reservation();
    if plan.sealed_bytes >= profile::SEALED_PEAK_BYTES {
        return Err("sealed reservation does not fit the 28.5 GiB peak".into());
    }
    Ok(Loaded { model_sha256: actual, donor_revision: revision.into(), reservation: plan })
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex(digest.finalize()))
}

pub fn sha256_path(source: &Path) -> Result<String, String> {
    let mut digest = Sha256::new();
    let root;
    let mut paths = Vec::new();
    if source.is_file() {
        root = source.parent().unwrap_or(Path::new(".")).to_path_buf();
        paths.push(source.to_path_buf());
    } else if source.is_dir() {
        root = source.to_path_buf();
        collect_files(source, &root, &mut paths)?;
    } else {
        return Err(format!("path not found: {}", source.display()));
    }
    paths.sort_by(|left, right| left.to_string_lossy().cmp(&right.to_string_lossy()));
    for item in paths {
        let relative = item.strip_prefix(&root).map_err(|error| error.to_string())?.to_string_lossy();
        let bytes = relative.as_bytes();
        digest.update(&(bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
        let size = fs::metadata(&item).map_err(|error| error.to_string())?.len();
        digest.update(&size.to_be_bytes());
        let mut file = fs::File::open(&item).map_err(|error| error.to_string())?;
        let mut buffer = [0u8; 1024 * 1024];
        loop {
            let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
    }
    Ok(hex(digest.finalize()))
}

fn collect_files(directory: &Path, root: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        if kind.is_dir() {
            collect_files(&path, root, out)?;
        } else if kind.is_file() {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let excluded = relative.components().any(|component| component.as_os_str() == ".cache" || component.as_os_str() == ".git");
            let name = relative.file_name().map(|name| name.to_string_lossy().to_string()).unwrap_or_default();
            let transient = name.ends_with(".part") || name.ends_with(".lock") || name.ends_with(".incomplete");
            if !excluded && !transient {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn directory_digest_matches_the_python_reference() {
        let root = std::env::temp_dir().join(format!("q38-sha256-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join(".cache")).unwrap();
        fs::create_dir_all(root.join("nested")).unwrap();
        fs::write(root.join("config.json"), b"{\"bits\": 5.0}").unwrap();
        fs::write(root.join("nested/shard.safetensors"), b"weights").unwrap();
        fs::write(root.join("partial.part"), b"ignored").unwrap();
        fs::write(root.join(".cache/ignored.bin"), b"ignored").unwrap();
        let digest = sha256_path(&root).unwrap();
        assert_eq!(digest, "4d36b74af0089375358a6716f7b4a77c626bd52f9b65c651b91070a2ad1be078");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_hash_mismatch_never_becomes_ready() {
        let root = std::env::temp_dir().join(format!("q38-artifact-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), br#"{"max_position_embeddings":262144,"quantization_config":{"quant_method":"exl3","bits":5.0}}"#).unwrap();
        let donor = root.join("donor");
        fs::create_dir_all(&donor).unwrap();
        fs::write(donor.join("shard.safetensors"), b"nvfp4").unwrap();
        let model_sha = sha256_path(&root).unwrap();
        let shard_sha = sha256_file(&donor.join("shard.safetensors")).unwrap();
        let model_manifest = json!({"model": {"artifact_sha256": model_sha}});
        let donor_manifest = json!({"revision": "abc", "shards": {"shard.safetensors": {"size": 5, "sha256": shard_sha}}});
        let loaded = verify(&root, &donor, &model_manifest, &donor_manifest).unwrap();
        assert!(!loaded.reservation.device_allocated);
        assert_eq!(loaded.reservation.kv_bytes, 4_831_838_208);
        let wrong = json!({"model": {"artifact_sha256": "0".repeat(64)}});
        assert!(verify(&root, &donor, &wrong, &donor_manifest).is_err_and(|error| error.contains("hash mismatch")));
        let _ = fs::remove_dir_all(root);
    }
}
