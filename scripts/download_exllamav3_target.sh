#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${QWASAR_MODEL_PATH:-/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw}"
REPOSITORY="thelastspark/Qwen3.8-27B-exl3"
REVISION="1a6fe4afb5b921fda9f93fd4b06d6c6d5c99a62c"

mkdir -p "$MODEL_DIR"

download_file() {
    local path="$1"
    local expected_sha256="$2"
    local destination="$MODEL_DIR/$path"
    local partial="$destination.part"
    if [[ -f "$destination" ]]; then
        if printf '%s  %s\n' "$expected_sha256" "$destination" | sha256sum --check --status; then
            printf 'already verified %s\n' "$path"
            return
        fi
        printf 'Existing file has the wrong checksum: %s\n' "$destination" >&2
        exit 2
    fi
    curl --location --fail --retry 8 --retry-all-errors --retry-delay 2 \
        --continue-at - --output "$partial" \
        "https://huggingface.co/$REPOSITORY/resolve/$REVISION/$path?download=true"
    printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum --check --status
    mv "$partial" "$destination"
    printf 'verified %s\n' "$path"
}

download_metadata() {
    local path="$1"
    if [[ ! -f "$MODEL_DIR/$path" ]]; then
        curl --location --fail --retry 8 --retry-all-errors --retry-delay 2 \
            --output "$MODEL_DIR/$path" \
            "https://huggingface.co/$REPOSITORY/resolve/$REVISION/$path?download=true"
    fi
}

for metadata in \
    .gitattributes LICENSE README.md chat_template.jinja config.json crc32.txt \
    generation_config.json merges.txt model.safetensors.index.json \
    preprocessor_config.json quantization_config.json tokenizer.json \
    tokenizer_config.json video_preprocessor_config.json vocab.json; do
    download_metadata "$metadata"
done

download_file model-00001-of-00003.safetensors \
    9c61af94100858ddd93b2195c1d529b8b0865a7f0c920e6b05a3c1fa44df4c6e &
pid1=$!
download_file model-00002-of-00003.safetensors \
    32fa6ee929a5aa1e9c9ae79d9c9b29cba5c8d023fcfd88be5ae57d927e1e5a91 &
pid2=$!
download_file model-00003-of-00003.safetensors \
    b24f78f172a524b74d896c07297623f01c1569e9c2ab49ab85a3c64d997d4fce &
pid3=$!
wait "$pid1" "$pid2" "$pid3"

printf 'Target downloaded and verified at %s\n' "$MODEL_DIR"
