#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array1, Array2, AsArray, Axis, Ix2, IxDyn, Zip};
use num_traits::Zero;

use crate::quantized::model::*;
use crate::{
    model_traits::*,
    util::{sigmoid, ParDot},
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
        let (tm_last_x, aa, bb, pp) = state.get_tm_state();

        let xk = &self.time.mix_k.mix(&x, tm_last_x);
        let xv = &self.time.mix_v.mix(&x, tm_last_x);
        let xr = &self.time.mix_r.mix(&x, tm_last_x);

        let r = sigmoid(self.receptance_weight.pardot(xr));
        let k = self.key_weight.pardot(xk);
        let v = self.value_weight.pardot(xv);

        let ww = &self.time.first + &k;
        let qq = Zip::from(&ww).and(pp).map_collect(|el1, el2| el1.max(*el2));
        let e1 = (pp - &qq).mapv(ATy::exp);
        let e2 = (ww - qq).mapv(ATy::exp);
        let a = &e1 * aa + &e2 * &v;
        let b = &e1 * bb + e2;

        let wkv = a / b;
        let ww = pp + &self.time.decay;
        let qq = Zip::from(&ww).and(&k).map_collect(|el1, el2| el1.max(*el2));
        let e1 = (ww - &qq).mapv(ATy::exp);
        let e2 = (k - &qq).mapv(ATy::exp);

        let new_aa = &e1 * aa + &e2 * v;
        let new_bb = e1 * bb + e2;
        let new_pp = qq;
        state.set_tm_state(x, new_aa, new_bb, new_pp);
        self.output_weight.pardot(&(r * wkv))
    }
}

impl RunFFN<ATy> for FeedForwardNetwork {
    type State = Array1<ATy>;
    fn channel_mixing<S: HasRWKVLayerState<ATy, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State {
        let cm_last_x = state.get_cm_state();
        let zero = ATy::zero();

        let xk = &self.time.mix_k.mix(&x, cm_last_x);
        let xr = &self.time.mix_r.mix(&x, cm_last_x);
        let r = sigmoid(self.receptance_weight.pardot(xr));

        let mut k = self.key_weight.pardot(xk);
        // ReLU + square
        k.mapv_inplace(|el| zero.max(el).powi(2));

        state.set_cm_state(x);
        r * self.value_weight.pardot(&k)
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

        self.head_weight.pardot(&self.ln_out.norm(&x))
    }
}
