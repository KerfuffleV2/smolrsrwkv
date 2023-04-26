use rusty_ggml::prelude::{GContext, GTensor1, GTensor2, GType};

// Corresponds to:
// 1. blocks.N.att.time_mix_[kvr]
// 2. blocks.N.ffn.time_mix_[kr]
pub struct Mix(pub GTensor1);

/// Corresponds to:
/// 1. ln_out.[bias,weight]
/// 2. blocks.N.ln[012].[bias,weight]
/// However, note that ln0 only exists in block 0.
pub struct LayerNorm {
    pub bias: GTensor1,
    pub weight: GTensor1,
}

/// Corresponds to:
/// 1. blocks.N.time_[first,decay]
/// 2. blocks.N.time_mix_[kvr]
pub struct AttTime {
    pub decay: GTensor1,
    pub mix_k: Mix,
    pub mix_v: Mix,
    pub mix_r: Mix,
    pub first: GTensor1,
}

/// Corresponds to:
/// 1. blocks.N.ffn.time_mix_[kr]
pub struct FFNTime {
    pub mix_k: Mix,
    pub mix_r: Mix,
}

/// Corresponds to:
/// 1. blocks.N.att.[key,value,output,receptance].weight
/// 3. Keys described in AttTime.
pub struct Attention {
    pub key_weight: GTensor2,
    pub value_weight: GTensor2,
    pub output_weight: GTensor2,
    pub receptance_weight: GTensor2,
    pub time: AttTime,
}

/// Corresponds to:
/// 1. blocks.N.ffn.[key,value,receptance].weight
/// 3. Keys described in FFNTime.
pub struct FeedForwardNetwork {
    pub key_weight: GTensor2,
    pub value_weight: GTensor2,
    pub receptance_weight: GTensor2,
    pub time: FFNTime,
}

/// See the comments for Attention, FeedForwardNetwork and LayerNorm.
pub struct RWKVLayer {
    /// Layer normalization used for time mixing (ln1).
    pub ln_tm: LayerNorm,
    /// Layer normalization used for channel mixing (ln2).
    pub ln_cm: LayerNorm,
    pub att: Attention,
    pub ffn: FeedForwardNetwork,
}

pub struct RWKV {
    pub ctx: GContext,
    pub emb: GTensor2,
    pub head_weight: GTensor2,
    pub ln_out: LayerNorm,
    // pub ln0: LayerNorm,
    pub layers: Vec<RWKVLayer>,

    /// Number of vocabulary items.
    pub n_vocab: usize,
    /// Number of embedding items.
    pub n_embed: usize,
    /// Number of layers in the model.
    pub n_layers: usize,
}

pub struct RWKVLayerState {
    pub tm_last_x: GTensor1,
    pub tm_aa: GTensor1,
    pub tm_bb: GTensor1,
    pub tm_pp: GTensor1,
    pub cm_last_x: GTensor1,
}

impl RWKVLayerState {
    pub fn new(ctx: &GContext, n_embed: usize) -> Self {
        let mut it = (0..5).map(|idx| {
            let mut t = ctx.tensor(GType::F32, [n_embed]);
            t.fill_f32(if idx != 4 { 0.0 } else { f32::NEG_INFINITY });
            t
        });
        Self {
            cm_last_x: it.next().unwrap(),
            tm_last_x: it.next().unwrap(),
            tm_aa: it.next().unwrap(),
            tm_bb: it.next().unwrap(),
            tm_pp: it.next().unwrap(),
        }
    }
}
