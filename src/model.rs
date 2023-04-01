#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array1, Array2};

use crate::util::ReqOps;

#[derive(Debug, Clone, PartialEq)]
/// Corresponds to:
/// 1. blocks.N.att.time_mix_[kvr]
/// 2. blocks.N.ffn.time_mix_[kr]
pub struct Mix<T>(pub Array1<T>);

#[derive(Debug, Clone, PartialEq)]
/// Corresponds to:
/// 1. ln_out.[bias,weight]
/// 2. blocks.N.ln[012].[bias,weight]
/// However, note that ln0 only exists in block 0.
pub struct LayerNorm<T> {
    pub bias: Array1<T>,
    pub weight: Array1<T>,
}

pub struct LayerNorm2<T> {
    pub bias: Array1<T>,
    pub weight: Array1<T>,
}

#[derive(Debug, Clone, PartialEq)]
/// Corresponds to:
/// 1. blocks.N.time_[first,decay]
/// 2. blocks.N.time_mix_[kvr]
pub struct AttTime<T> {
    pub decay: Array1<T>,
    pub mix_k: Mix<T>,
    pub mix_v: Mix<T>,
    pub mix_r: Mix<T>,
    pub first: Array1<T>,
}

/// Corresponds to:
/// 1. blocks.N.ffn.time_mix_[kr]
#[derive(Debug, Clone, PartialEq)]
pub struct FFNTime<T> {
    pub mix_k: Mix<T>,
    pub mix_r: Mix<T>,
}

/// Corresponds to:
/// 1. blocks.N.att.[key,value,output,receptance].weight
/// 3. Keys described in AttTime.
#[derive(Debug, Clone, PartialEq)]
pub struct Attention<T> {
    pub key_weight: Array2<T>,
    pub value_weight: Array2<T>,
    pub output_weight: Array2<T>,
    pub receptance_weight: Array2<T>,
    pub time: AttTime<T>,
}

/// Corresponds to:
/// 1. blocks.N.ffn.[key,value,receptance].weight
/// 3. Keys described in FFNTime.
#[derive(Debug, Clone, PartialEq)]
pub struct FeedForwardNetwork<T> {
    pub key_weight: Array2<T>,
    pub value_weight: Array2<T>,
    pub receptance_weight: Array2<T>,
    pub time: FFNTime<T>,
}

#[derive(Debug, Clone, PartialEq)]
/// See the comments for Attention, FeedForwardNetwork and LayerNorm.
pub struct RWKVLayer<T> {
    /// l1 is used for time mixing,
    pub ln1: LayerNorm<T>,
    /// ln2 is used for channel mixing.
    pub ln2: LayerNorm<T>,
    pub att: Attention<T>,
    pub ffn: FeedForwardNetwork<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RWKV<T> {
    /// emb.weight
    pub emb: Array2<T>,
    /// head.weight
    pub head: Array2<T>,
    /// ln_out.[weight,bias]
    pub ln_out: LayerNorm<T>,
    pub layers: Vec<RWKVLayer<T>>,
}

#[derive(Clone, PartialEq)]
/// Each layer has its own independent state.
pub struct RWKVLayerState<T> {
    /// State from time mixing.
    pub tm_last_x: Array1<T>,
    /// Time mixing numerator?
    pub tm_num: Array1<T>,
    /// Time mixing denominator?
    pub tm_den: Array1<T>,
    /// State from channel mixing.
    pub cm_last_x: Array1<T>,
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
