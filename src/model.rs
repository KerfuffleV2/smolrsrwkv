#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array1, Array2, Ix1};

use crate::util::{FloatTensor, ReqOps};

#[derive(Debug, Clone, PartialEq)]
/// Corresponds to:
/// 1. blocks.N.att.time_mix_[kvr]
/// 2. blocks.N.ffn.time_mix_[kr]
pub struct Mix<WT>(pub Array1<WT>);

#[derive(Debug, Clone, PartialEq)]
/// Corresponds to:
/// 1. ln_out.[bias,weight]
/// 2. blocks.N.ln[012].[bias,weight]
/// However, note that ln0 only exists in block 0.
pub struct LayerNorm<WT> {
    pub bias: Array1<WT>,
    pub weight: Array1<WT>,
}

#[derive(Debug, Clone, PartialEq)]
/// Corresponds to:
/// 1. blocks.N.time_[first,decay]
/// 2. blocks.N.time_mix_[kvr]
pub struct AttTime<WT, AT> {
    pub decay: Array1<AT>,
    pub mix_k: Mix<WT>,
    pub mix_v: Mix<WT>,
    pub mix_r: Mix<WT>,
    pub first: FloatTensor<AT>,
}

/// Corresponds to:
/// 1. blocks.N.ffn.time_mix_[kr]
#[derive(Debug, Clone, PartialEq)]
pub struct FFNTime<WT> {
    pub mix_k: Mix<WT>,
    pub mix_r: Mix<WT>,
}

/// Corresponds to:
/// 1. blocks.N.att.[key,value,output,receptance].weight
/// 3. Keys described in AttTime.
#[derive(Debug, Clone, PartialEq)]
pub struct Attention<WT, AT> {
    pub key_weight: Array2<WT>,
    pub value_weight: Array2<WT>,
    pub output_weight: Array2<WT>,
    pub receptance_weight: Array2<WT>,
    pub time: AttTime<WT, AT>,
}

/// Corresponds to:
/// 1. blocks.N.ffn.[key,value,receptance].weight
/// 3. Keys described in FFNTime.
#[derive(Debug, Clone, PartialEq)]
pub struct FeedForwardNetwork<WT, AT> {
    pub key_weight: Array2<WT>,
    pub value_weight: Array2<WT>,
    pub receptance_weight: Array2<WT>,
    pub time: FFNTime<AT>,
}

#[derive(Debug, Clone, PartialEq)]
/// See the comments for Attention, FeedForwardNetwork and LayerNorm.
pub struct RWKVLayer<WT, AT> {
    /// l1 is used for time mixing,
    pub ln1: LayerNorm<WT>,
    /// ln2 is used for channel mixing.
    pub ln2: LayerNorm<WT>,
    pub att: Attention<WT, AT>,
    pub ffn: FeedForwardNetwork<WT, AT>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RWKV<WT, AT> {
    /// emb.weight
    pub emb: Array2<WT>,
    /// head.weight
    pub head: Array2<WT>,
    /// ln_out.[weight,bias]
    pub ln_out: LayerNorm<WT>,
    pub layers: Vec<RWKVLayer<WT, AT>>,

    /// Number of vocabulary items.
    pub n_vocab: usize,
    /// Number of embedding items.
    pub n_embed: usize,
    /// Number of layers in the model.
    pub n_layers: usize,
}

#[derive(Clone, PartialEq)]
/// Each layer has its own independent state.
pub struct RWKVLayerState<WT> {
    /// State from time mixing.
    pub tm_last_x: Array1<WT>,
    /// Time mixing numerator?
    pub tm_num: Array1<WT>,
    /// Time mixing denominator?
    pub tm_den: Array1<WT>,
    /// State from channel mixing.
    pub cm_last_x: Array1<WT>,
}

impl<T: ReqOps> RWKVLayerState<T> {
    pub fn new(n_embed: usize) -> Self {
        let zs = Array1::zeros(n_embed);
        Self {
            tm_last_x: zs.clone(),
            tm_num: zs.clone(),
            tm_den: zs.clone(),
            cm_last_x: zs,
        }
    }
}
