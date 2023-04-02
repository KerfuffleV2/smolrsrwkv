#![allow(clippy::upper_case_acronyms)]
#![allow(unused_imports, dead_code, unused_variables)]
use ndarray::{Array1, Array2, ArrayView1, AsArray, Axis, IntoDimension, Ix2, IxDyn};
use num_traits::{Bounded, One, Zero};

use crate::quantized::model::*;
use crate::simple::{model as S, model_impls as SI};
use crate::{
    model_traits::*,
    rwkvops::RWKVOps11,
    util::{pardot, pardot8, ReqOps},
};

fn amin<'a, A: AsArray<'a, ATy, Ix2>>(arr: A, axis: Axis) -> Array1<ATy> {
    arr.into()
        .axis_iter(axis)
        .map(|a| a.iter().copied().fold(ATy::INFINITY, |a, b| a.min(b)))
        .collect::<Array1<ATy>>()
}

fn amax<'a, A: AsArray<'a, ATy, Ix2>>(arr: A, axis: Axis) -> Array1<ATy> {
    arr.into()
        .axis_iter(axis)
        .map(|a| a.iter().copied().fold(ATy::NEG_INFINITY, |a, b| a.max(b)))
        .collect::<Array1<ATy>>()
}

impl From<Array2<ATy>> for TensorQ2 {
    fn from(mut value: Array2<ATy>) -> Self {
        let shape = value.shape();
        let (mx, my) = if shape[0] > shape[1] {
            let miny = amin(&value, Axis(0)).insert_axis(Axis(1));
            value -= &miny;
            let miny = miny
                .into_dimensionality::<IxDyn>()
                .expect("miny failed dimensionality conversion!");
            let minx = amin(&value, Axis(1));
            value -= &minx;
            let minx = minx
                .into_dimensionality::<IxDyn>()
                .expect("minx failed dimensionality conversion!");
            (minx, miny)
        } else {
            let miny = amin(&value, Axis(1));
            value -= &miny;
            let miny = miny
                .into_dimensionality::<IxDyn>()
                .expect("miny failed dimensionality conversion!");
            let minx = amin(&value, Axis(0)).insert_axis(Axis(1));
            value -= &minx;
            let minx = minx
                .into_dimensionality::<IxDyn>()
                .expect("minx failed dimensionality conversion!");

            (minx, miny)
        };
        let rx = amax(&value, Axis(1));
        value /= &rx;
        let ry = amax(&value, Axis(0)).insert_axis(Axis(1));
        value /= &ry;
        let weight = value.mapv(|el| (el * 256.0).floor().clamp(0.0, 255.0) as u8);
        Self {
            weight,
            mx,
            my,
            rx: rx / 16.0,
            ry: ry / 16.0,
        }
    }
}

impl RunAttention<ATy> for Attention {
    type State = Array1<ATy>;
    fn time_mixing<S: HasRWKVLayerState<ATy, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State {
        let (last_x, last_num, last_den) = state.get_tm_state();

        let mix_k = &self.time.mix_k.mix(&x, last_x);
        let mix_v = &self.time.mix_v.mix(&x, last_x);
        let mix_r = &self.time.mix_r.mix(&x, last_x);

        let k = pardot8(&self.key_weight, mix_k);
        let v = pardot8(&self.value_weight, mix_v);
        let r = pardot8(&self.receptance_weight, mix_r);

        let exp_k = k.mapv(|el| el.exp());
        let exp_decay = self.time.decay.view();

        let wkv = {
            let e = (&self.time.first + &k).mapv(|el| el.exp());
            (last_num + (&e * &v)) / (last_den + e)
        };
        let rwkv = r.sigmoid() * wkv;

        let num = (&exp_decay * last_num) + (&exp_k * &v);
        let den = (&exp_decay * last_den) + &exp_k;
        state.set_tm_state(x, num, den);
        pardot8(&self.output_weight, &rwkv)
    }
}

impl RunFFN<ATy> for FeedForwardNetwork {
    type State = Array1<ATy>;
    fn channel_mixing<S: HasRWKVLayerState<ATy, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State {
        let last_x = state.get_cm_state();

        let mix_k = &self.time.mix_k.mix(&x, last_x);
        let mix_r = &self.time.mix_r.mix(&x, last_x);

        let k = pardot8(&self.key_weight, mix_k);
        let r = pardot8(&self.receptance_weight, mix_r);
        let vk = pardot8(
            &self.value_weight,
            &k.mapv(|val| val.max(ATy::zero()).powi(2)),
        );

        state.set_cm_state(x);
        r.sigmoid() * &vk
    }
}

impl RunRWKVLayer<ATy> for RWKVLayer {
    type XTy = Array1<ATy>;
    type Out = Array1<ATy>;

    fn evaluate<S: HasRWKVLayerState<ATy, State = Self::Out>>(
        &self,
        x: Self::XTy,
        state: &mut S,
    ) -> Self::Out {
        let x = self.att.time_mixing(self.ln_tm.norm(&x), state) + &x;
        self.ffn.channel_mixing(self.ln_cm.norm(&x), state) + &x
    }
}

impl RunRWKV<ATy> for RWKV {
    type Out = Array1<ATy>;
    type Token = usize;

    fn evaluate<S: HasRWKVLayerState<ATy, State = Self::Out>>(
        &self,
        token: Self::Token,
        state: &mut [S],
    ) -> Self::Out {
        let initial_x = self.emb.index_axis(Axis(0), token).to_owned();

        let x = self
            .layers
            .iter()
            .enumerate()
            .fold(initial_x, |x, (lnum, layer)| {
                layer.evaluate(x, &mut state[lnum])
            });

        let x = pardot8(&self.head, &self.ln_out.norm(&x));
        let x_max = x.fold(ATy::min_value(), |acc, el| acc.max(*el));
        let e_x = (x - x_max).mapv(|el| el.exp());

        &e_x / e_x.sum()
    }
}
