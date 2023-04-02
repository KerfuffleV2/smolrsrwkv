#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array2, ArrayD};

use crate::simple::model as S;

/// Attention type (and for items that need a full float).
pub type ATy = f32;
/// Quantized type.
pub type WTy = u8;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorQ2 {
    pub weight: Array2<WTy>,
    pub mxy: ArrayD<ATy>,
    pub ryx: Array2<ATy>,
}

/// Corresponds to:
/// 1. blocks.N.att.[key,value,output,receptance].weight
/// 3. Keys described in AttTime.
#[derive(Debug, Clone, PartialEq)]
pub struct Attention {
    pub key_weight: TensorQ2,
    pub value_weight: TensorQ2,
    pub output_weight: TensorQ2,
    pub receptance_weight: TensorQ2,
    pub time: S::AttTime<ATy>,
}

/// Corresponds to:
/// 1. blocks.N.ffn.[key,value,receptance].weight
/// 3. Keys described in FFNTime.
#[derive(Debug, Clone, PartialEq)]
pub struct FeedForwardNetwork {
    pub key_weight: TensorQ2,
    pub value_weight: TensorQ2,
    pub receptance_weight: TensorQ2,
    pub time: S::FFNTime<ATy>,
}

#[derive(Debug, Clone, PartialEq)]
/// See the comments for Attention, FeedForwardNetwork and LayerNorm.
pub struct RWKVLayer {
    /// Layer normalization used for time mixing (ln1).
    pub ln_tm: S::LayerNorm<ATy>,
    /// Layer normalization used for channel mixing (ln2).
    pub ln_cm: S::LayerNorm<ATy>,
    pub att: Attention,
    pub ffn: FeedForwardNetwork,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RWKV {
    /// emb.weight
    pub emb: Array2<ATy>,
    /// head.weight
    pub head: TensorQ2,
    /// ln_out.[weight,bias]
    pub ln_out: S::LayerNorm<ATy>,
    pub layers: Vec<RWKVLayer>,

    /// Number of vocabulary items.
    pub n_vocab: usize,
    /// Number of embedding items.
    pub n_embed: usize,
    /// Number of layers in the model.
    pub n_layers: usize,
}
