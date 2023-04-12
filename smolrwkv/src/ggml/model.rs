use ggml::{Context, Tensor, Type as GT};

// Corresponds to:
// 1. blocks.N.att.time_mix_[kvr]
// 2. blocks.N.ffn.time_mix_[kr]
pub struct Mix(pub Tensor);

/// Corresponds to:
/// 1. ln_out.[bias,weight]
/// 2. blocks.N.ln[012].[bias,weight]
/// However, note that ln0 only exists in block 0.
pub struct LayerNorm {
    pub bias: Tensor,
    pub weight: Tensor,
}

/// Corresponds to:
/// 1. blocks.N.time_[first,decay]
/// 2. blocks.N.time_mix_[kvr]
pub struct AttTime {
    pub decay: Tensor,
    pub mix_k: Mix,
    pub mix_v: Mix,
    pub mix_r: Mix,
    pub first: Tensor,
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
    pub key_weight: Tensor,
    pub value_weight: Tensor,
    pub output_weight: Tensor,
    pub receptance_weight: Tensor,
    pub time: AttTime,
}

/// Corresponds to:
/// 1. blocks.N.ffn.[key,value,receptance].weight
/// 3. Keys described in FFNTime.
pub struct FeedForwardNetwork {
    pub key_weight: Tensor,
    pub value_weight: Tensor,
    pub receptance_weight: Tensor,
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
    pub ctx: Context,
    pub emb: Tensor,
    pub head_weight: Tensor,
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
    pub tm_last_x: Tensor,
    pub tm_aa: Tensor,
    pub tm_bb: Tensor,
    pub tm_pp: Tensor,
    pub cm_last_x: Tensor,
}

impl RWKVLayerState {
    pub fn new(ctx: &Context, n_embed: usize) -> Self {
        let cm_last_x = ctx.new_tensor_1d(GT::F32, n_embed);
        let tm_last_x = ctx.new_tensor_1d(GT::F32, n_embed);
        let tm_aa = ctx.new_tensor_1d(GT::F32, n_embed);
        let tm_bb = ctx.new_tensor_1d(GT::F32, n_embed);

        cm_last_x.zero_data();
        tm_last_x.zero_data();
        tm_aa.zero_data();
        tm_bb.zero_data();

        // FIXME: Better way?
        let tm_pp = ctx.new_tensor_1d(GT::F32, n_embed);
        unsafe {
            let d = std::slice::from_raw_parts_mut(tm_pp.data() as *mut f32, n_embed);
            d.iter_mut().for_each(|d| *d = f32::NEG_INFINITY);
        }
        Self {
            cm_last_x,
            tm_last_x,
            tm_aa,
            tm_bb,
            tm_pp,
        }
    }
}
