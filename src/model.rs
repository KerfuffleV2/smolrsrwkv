#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::util::{pardot, sigmoid, ReqOps};

#[derive(Debug, Clone, PartialEq)]
pub struct Mix<T>(pub Array1<T>);

#[derive(Debug, Clone, PartialEq)]
pub struct LayerNorm<T> {
    pub bias: Array1<T>,
    pub weight: Array1<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttTime<T> {
    pub decay: Array1<T>,
    pub mix_k: Mix<T>,
    pub mix_v: Mix<T>,
    pub mix_r: Mix<T>,
    pub first: Array1<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FFNTime<T> {
    pub mix_k: Mix<T>,
    pub mix_r: Mix<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attention<T> {
    pub key_weight: Array2<T>,
    pub value_weight: Array2<T>,
    pub output_weight: Array2<T>,
    pub receptance_weight: Array2<T>,
    pub time: AttTime<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeedForwardNetwork<T> {
    pub key_weight: Array2<T>,
    pub value_weight: Array2<T>,
    pub receptance_weight: Array2<T>,
    pub time: FFNTime<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer<T> {
    pub ln: [LayerNorm<T>; 2],
    pub att: Attention<T>,
    pub ffn: FeedForwardNetwork<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RWKV<T> {
    pub emb: Array2<T>,
    pub head: Array2<T>,
    pub ln_out: LayerNorm<T>,
    pub ln0: LayerNorm<T>,
    pub layers: Vec<Layer<T>>,
}

#[derive(Clone, PartialEq)]
pub struct RWKVLayerState<T> {
    pub tm_state: Array1<T>,
    pub tm_num: Array1<T>,
    pub tm_den: Array1<T>,
    pub cm_state: Array1<T>,
}

impl<T: ReqOps> RWKVLayerState<T> {
    pub fn new(n_embed: usize) -> Self {
        let zs = Array1::zeros(n_embed);
        Self {
            tm_state: zs.clone(),
            tm_num: zs.clone(),
            tm_den: zs.clone(),
            cm_state: zs,
        }
    }

    pub fn update(
        &mut self,
        tm_state: Array1<T>,
        tm_num: Array1<T>,
        tm_den: Array1<T>,
        cm_state: Array1<T>,
    ) {
        *self = Self {
            tm_state,
            tm_num,
            tm_den,
            cm_state,
        }
    }
}

impl<T: ReqOps> Mix<T> {
    pub fn mix(&self, x: &ArrayView1<T>, last_x: &ArrayView1<T>) -> Array1<T> {
        x * &self.0 + last_x * (T::one() - &self.0)
    }
}

impl<T: ReqOps> Attention<T> {
    pub fn time_mixing(
        &self,
        x: &ArrayView1<T>,
        state: &RWKVLayerState<T>,
    ) -> (Array1<T>, (Array1<T>, Array1<T>)) {
        let last_x = &state.tm_state.view();
        let last_num = &state.tm_num.view();
        let last_den = &state.tm_den.view();

        let k = pardot(&self.key_weight, &self.time.mix_k.mix(x, last_x));
        let v = pardot(&self.value_weight, &self.time.mix_v.mix(x, last_x));
        let r = pardot(&self.receptance_weight, &self.time.mix_r.mix(x, last_x));

        let exp_k = k.mapv(|el| el.exp());
        let exp_decay = self.time.decay.mapv(|el| (-el.exp()).exp());

        let wkv = {
            let e = (&self.time.first + &k).mapv(|el| el.exp());
            (last_num + &e * &v) / (last_den + e)
        };
        let rwkv = sigmoid(&r) * wkv;

        let num = &exp_decay * last_num + &exp_k * &v;
        let den = &exp_decay * last_den + &exp_k;
        (pardot(&self.output_weight, &rwkv), (num, den))
    }
}

impl<T: ReqOps> FeedForwardNetwork<T> {
    pub fn channel_mixing(&self, x: &ArrayView1<T>, state: &RWKVLayerState<T>) -> Array1<T> {
        let last_x = &state.cm_state.view();
        let k = pardot(&self.key_weight, &self.time.mix_k.mix(x, last_x));
        let r = pardot(&self.receptance_weight, &self.time.mix_r.mix(x, last_x));
        let vk = pardot(
            &self.value_weight,
            &k.mapv(|val| val.max(T::zero()).powi(2)),
        );

        sigmoid(&r) * &vk
    }
}

impl<T: ReqOps> LayerNorm<T> {
    pub fn norm(&self, x: &ArrayView1<T>) -> Array1<T> {
        let mean = x.mean().unwrap();
        let std = x.std(T::zero());
        (x - mean) / std * &self.weight + &self.bias
    }
}

impl<T: ReqOps> RWKV<T> {
    pub fn evaluate_layer(
        &self,
        mut x: Array1<T>,
        layer: &Layer<T>,
        layer_state: &mut RWKVLayerState<T>,
    ) -> Array1<T> {
        let x_ln1 = layer.ln[0].norm(&x.view());
        let (dx, (tm_num, tm_den)) = layer.att.time_mixing(&x_ln1.view(), layer_state);
        x += &dx;

        let x_ln2 = layer.ln[1].norm(&x.view());
        let dx = layer.ffn.channel_mixing(&x_ln2.view(), layer_state);
        x += &dx;

        layer_state.update(x_ln1, tm_num, tm_den, x_ln2);
        x
    }

    pub fn evaluate(&self, token: usize, state: &mut [RWKVLayerState<T>]) -> Array1<T> {
        let initial_x = self.ln0.norm(&self.emb.index_axis(Axis(0), token));

        let x = self
            .layers
            .iter()
            .enumerate()
            .fold(initial_x, |x, (lnum, layer)| {
                self.evaluate_layer(x, layer, &mut state[lnum])
            });

        let x = pardot(&self.head, &self.ln_out.norm(&x.view()));
        let x_max = x.fold(T::min_value(), |acc, el| acc.max(*el));
        let e_x = (x - x_max).mapv(|el| el.exp());

        &e_x / e_x.sum()
    }
}
