use rusty_ggml::{
    context::GgmlContext as Context,
    tensor::{GgmlElementType as GT, GgmlTensor as Tensor},
};

// Corresponds to:
// 1. blocks.N.att.time_mix_[kvr]
// 2. blocks.N.ffn.time_mix_[kr]
pub struct Mix(pub Tensor<1>);

/// Corresponds to:
/// 1. ln_out.[bias,weight]
/// 2. blocks.N.ln[012].[bias,weight]
/// However, note that ln0 only exists in block 0.
pub struct LayerNorm {
    pub bias: Tensor<1>,
    pub weight: Tensor<1>,
}

/// Corresponds to:
/// 1. blocks.N.time_[first,decay]
/// 2. blocks.N.time_mix_[kvr]
pub struct AttTime {
    pub decay: Tensor<1>,
    pub mix_k: Mix,
    pub mix_v: Mix,
    pub mix_r: Mix,
    pub first: Tensor<1>,
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
    pub key_weight: Tensor<2>,
    pub value_weight: Tensor<2>,
    pub output_weight: Tensor<2>,
    pub receptance_weight: Tensor<2>,
    pub time: AttTime,
}

/// Corresponds to:
/// 1. blocks.N.ffn.[key,value,receptance].weight
/// 3. Keys described in FFNTime.
pub struct FeedForwardNetwork {
    pub key_weight: Tensor<2>,
    pub value_weight: Tensor<2>,
    pub receptance_weight: Tensor<2>,
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
    pub emb: Tensor<2>,
    pub head_weight: Tensor<2>,
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
    pub tm_last_x: Tensor<1>,
    pub tm_aa: Tensor<1>,
    pub tm_bb: Tensor<1>,
    pub tm_pp: Tensor<1>,
    pub cm_last_x: Tensor<1>,
}

impl RWKVLayerState {
    pub fn new(ctx: &Context, n_embed: usize) -> Self {
        let mut cm_last_x = ctx.tensor(GT::F32, [n_embed]);
        let mut tm_last_x = ctx.tensor(GT::F32, [n_embed]);
        let mut tm_aa = ctx.tensor(GT::F32, [n_embed]);
        let mut tm_bb = ctx.tensor(GT::F32, [n_embed]);
        let mut tm_pp = ctx.tensor(GT::F32, [n_embed]);

        // FIXME: This is pretty nasty.
        unsafe {
            cm_last_x.with_data_mut(|d| d.iter_mut().for_each(|dst| *dst = 0));
            tm_last_x.with_data_mut(|d| d.iter_mut().for_each(|dst| *dst = 0));
            tm_aa.with_data_mut(|d| d.iter_mut().for_each(|dst| *dst = 0));
            tm_bb.with_data_mut(|d| d.iter_mut().for_each(|dst| *dst = 0));
            tm_pp.with_data_mut(|d| {
                let s = std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut f32, n_embed);
                s.iter_mut().for_each(|dst| *dst = f32::NEG_INFINITY)
            })
        };

        Self {
            cm_last_x,
            tm_last_x,
            tm_aa,
            tm_bb,
            tm_pp,
        }
    }
}
