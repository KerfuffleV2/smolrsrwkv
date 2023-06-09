#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array1, ArrayView1, Axis, Zip};
use tracing::{instrument, span, Level};

use crate::simple::model::*;
use crate::{
    model_traits::*,
    util::{sigmoid, ParDot, ReqOps},
};

impl<T: ReqOps> HasRWKVLayerState<T> for RWKVLayerState<T> {
    type State = Array1<T>;

    /// Get time mixing stating as (last_x, aa, bb, pp).
    /// What does aa, bb, pp mean? Who knows!
    fn get_tm_state(&self) -> (&Self::State, &Self::State, &Self::State, &Self::State) {
        (&self.tm_last_x, &self.tm_aa, &self.tm_bb, &self.tm_pp)
    }

    /// Set time mixing state.
    fn set_tm_state(
        &mut self,
        tm_last_x: Self::State,
        aa: Self::State,
        bb: Self::State,
        pp: Self::State,
    ) {
        self.tm_last_x = tm_last_x;
        self.tm_aa = aa;
        self.tm_bb = bb;
        self.tm_pp = pp;
    }

    /// Get channel mixing state.
    fn get_cm_state(&self) -> &Self::State {
        &self.cm_last_x
    }

    /// Set channel mixing state.
    fn set_cm_state(&mut self, cm_last_x: Self::State) {
        self.cm_last_x = cm_last_x;
    }
}

impl<T: ReqOps> RunMix for Mix<T> {
    type XTy<'a> = ArrayView1<'a, T>;
    type Out = Array1<T>;

    fn mix<'a, X: Into<Self::XTy<'a>>, LX: Into<Self::XTy<'a>>>(
        &self,
        x: X,
        last_x: LX,
    ) -> Self::Out {
        let (x, time_mix, last_x) = (&x.into(), &self.0, &last_x.into());
        span!(Level::DEBUG, "mix", x = ?x.shape(), last_x = ?last_x.shape());
        x * time_mix + last_x * (T::one() - time_mix)
    }
}

impl<T: ReqOps, WT: ParDot<Output = Array1<T>>> RunAttention<T> for Attention<T, WT> {
    type State = Array1<T>;

    #[instrument(skip_all, level = "DEBUG")]
    fn time_mixing<S: HasRWKVLayerState<T, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State {
        let (tm_last_x, aa, bb, pp) = state.get_tm_state();

        let (ref xk, ref xv, ref xr) = span!(Level::DEBUG, "mixing").in_scope(|| {
            let xk = self.time.mix_k.mix(&x, tm_last_x);
            let xv = self.time.mix_v.mix(&x, tm_last_x);
            let xr = self.time.mix_r.mix(&x, tm_last_x);
            (xk, xv, xr)
        });

        let (r, k, v) = span!(Level::DEBUG, "rkv").in_scope(|| {
            let r = sigmoid(self.receptance_weight.pardot(xr));
            let k = self.key_weight.pardot(xk);
            let v = self.value_weight.pardot(xv);
            (r, k, v)
        });

        let (a, b) = span!(Level::DEBUG, "remaining-phase1").in_scope(|| {
            let ww = &self.time.first + &k;
            let qq = Zip::from(&ww).and(pp).map_collect(|el1, el2| el1.max(*el2));
            let e1 = (pp - &qq).mapv(T::exp);
            let e2 = (ww - qq).mapv(T::exp);
            let a = &e1 * aa + &e2 * &v;
            let b = &e1 * bb + e2;
            (a, b)
        });

        let (wkv, new_aa, new_bb, new_pp) =
            span!(Level::DEBUG, "remaining-phase2").in_scope(|| {
                let ww = pp + &self.time.decay;
                let qq = Zip::from(&ww).and(&k).map_collect(|el1, el2| el1.max(*el2));
                let e1 = (ww - &qq).mapv(T::exp);
                let e2 = (k - &qq).mapv(T::exp);

                let new_aa = &e1 * aa + &e2 * v;
                let new_bb = e1 * bb + e2;
                let new_pp = qq;

                (a / b, new_aa, new_bb, new_pp)
            });
        state.set_tm_state(x, new_aa, new_bb, new_pp);
        span!(Level::DEBUG, "output");
        self.output_weight.pardot(&(r * wkv))
    }
}

impl<T: ReqOps, WT: ParDot<Output = Array1<T>>> RunFFN<T> for FeedForwardNetwork<T, WT> {
    type State = Array1<T>;

    #[instrument(level = "DEBUG", skip_all)]
    fn channel_mixing<S: HasRWKVLayerState<T, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State {
        let cm_last_x = state.get_cm_state();
        let zero = T::zero();

        let (ref xk, ref xr) = span!(Level::DEBUG, "mixing").in_scope(|| {
            (
                self.time.mix_k.mix(&x, cm_last_x),
                self.time.mix_r.mix(&x, cm_last_x),
            )
        });

        let r = span!(Level::DEBUG, "r").in_scope(|| sigmoid(self.receptance_weight.pardot(xr)));

        let mut k = span!(Level::DEBUG, "k").in_scope(|| self.key_weight.pardot(xk));
        // ReLU + square
        span!(Level::DEBUG, "k-relu-squared")
            .in_scope(|| k.mapv_inplace(|el| zero.max(el).powi(2)));

        state.set_cm_state(x);
        span!(Level::DEBUG, "output");
        r * self.value_weight.pardot(&k)
    }
}

impl<T: ReqOps> RunLayerNorm for LayerNorm<T> {
    type XTy<'a> = ArrayView1<'a, T>;
    type Out = Array1<T>;

    #[instrument(skip_all, level = "DEBUG", name = "layer_norm")]
    fn norm<'a, X: Into<Self::XTy<'a>>>(&self, x: X) -> Self::Out {
        let origx = x.into();
        let x = &origx.view();
        let mean = x.mean().expect("Invalid valid in mean()");
        let std = x.std(T::zero());
        (((x - mean) / std) * &self.weight) + &self.bias
    }
}

impl<T: ReqOps, WT: ParDot<Output = Array1<T>>> RunRWKVLayer<T> for RWKVLayer<T, WT> {
    type XTy = Array1<T>;
    type Out = Array1<T>;

    #[instrument(skip_all, level = "DEBUG", name = "evaluate_layer")]
    fn evaluate<S: HasRWKVLayerState<T, State = Self::Out>>(
        &self,
        x: Self::XTy,
        state: &mut S,
    ) -> Self::Out {
        let x = self.att.time_mixing(self.ln_tm.norm(&x), state) + &x;
        self.ffn.channel_mixing(self.ln_cm.norm(&x), state) + &x
    }
}

impl<T: ReqOps, WT: ParDot<Output = Array1<T>>> RunRWKV<T> for RWKV<T, WT> {
    type Out = Array1<T>;
    type Token = usize;

    #[instrument(skip_all, level = "DEBUG", fields(token = token))]
    fn evaluate<S: HasRWKVLayerState<T, State = Self::Out>>(
        &self,
        token: Self::Token,
        state: &mut [S],
    ) -> Self::Out {
        let initial_x = self.emb.index_axis(Axis(0), token).to_owned();

        let x = span!(Level::DEBUG, "layers").in_scope(|| {
            self.layers
                .iter()
                .enumerate()
                .fold(initial_x, |x, (lnum, layer)| {
                    layer.evaluate(x, &mut state[lnum])
                })
        });

        span!(Level::DEBUG, "output");
        self.head_weight.pardot(&self.ln_out.norm(&x))
    }
}
