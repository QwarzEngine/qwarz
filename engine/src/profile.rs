//! Frozen Qwen3.8-27B profile for one RTX 5090.
//!
//! The numbers are the model, not a configuration. 48 Gated DeltaNet layers
//! and 16 full-attention layers alternate as three recurrent layers plus one
//! attention layer. The KV budget is the NVFP4 cache for the native 262,144
//! positions. Vision is resident; this is a single-stream engine.

pub const LAYERS: u32 = 64;
pub const GDN_LAYERS: u32 = 48;
pub const ATTENTION_LAYERS: u32 = 16;
pub const HIDDEN: u32 = 5_120;
pub const INTERMEDIATE: u32 = 17_408;
pub const QUERY_HEADS: u32 = 24;
pub const KV_HEADS: u32 = 4;
pub const HEAD_DIM: u32 = 256;
pub const CONTEXT_TOKENS: u64 = 262_144;
pub const VOCAB: u32 = 248_320;

pub const GROUP: u32 = 4;
pub const GDN_PER_GROUP: u32 = 3;

/// NVFP4 KV payload: 4.5 bits per element, K and V, 16 layers.
pub const fn kv_bytes(tokens: u64) -> u64 {
    ATTENTION_LAYERS as u64 * 2 * tokens * KV_HEADS as u64 * HEAD_DIM as u64 * 9 / 16
}

pub const WEIGHT_BUDGET_BYTES: u64 = 19 * 1024 * 1024 * 1024;
pub const GDN_STATE_BYTES: u64 = 144 * 1024 * 1024;
pub const VISION_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;
pub const DEVICE_BUDGET_BYTES: u64 = 30 * 1024 * 1024 * 1024;
pub const SEALED_PEAK_BYTES: u64 = (285 * 1024 * 1024 * 1024) / 10;

pub const fn sealed_bytes(tokens: u64) -> u64 {
    kv_bytes(tokens) + WEIGHT_BUDGET_BYTES + GDN_STATE_BYTES + VISION_BUDGET_BYTES
}

pub const fn layer_kind(index: u32) -> &'static str {
    if index % GROUP == GDN_PER_GROUP { "attention" } else { "gdn" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stack_is_three_recurrent_layers_then_one_attention_layer() {
        assert_eq!(GDN_LAYERS + ATTENTION_LAYERS, LAYERS);
        assert_eq!(LAYERS / GROUP, ATTENTION_LAYERS);
        let mut gdn = 0;
        let mut attention = 0;
        for index in 0..LAYERS {
            match layer_kind(index) {
                "gdn" => gdn += 1,
                "attention" => attention += 1,
                other => panic!("unexpected layer kind {other}"),
            }
        }
        assert_eq!((gdn, attention), (GDN_LAYERS, ATTENTION_LAYERS));
        assert_eq!(layer_kind(3), "attention");
        assert_eq!(layer_kind(63), "attention");
    }

    #[test]
    fn full_context_kv_is_four_and_a_half_gibibytes() {
        assert_eq!(kv_bytes(CONTEXT_TOKENS), 4_831_838_208);
        assert!(sealed_bytes(CONTEXT_TOKENS) < SEALED_PEAK_BYTES);
        assert!(SEALED_PEAK_BYTES < DEVICE_BUDGET_BYTES);
    }
}
