#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::{
    model::*,
    model_traits::*,
    rwkvops::RWKVOps11,
    util::{pardot, ReqOps},
};

impl<T: ReqOps> HasRWKVLayerState<T> for RWKVLayerState<T> {
    type State = Array1<T>;

    /// Get time mixing stating as (last_x, num, den).
    fn get_tm_state(&self) -> (&Self::State, &Self::State, &Self::State) {
        (&self.tm_last_x, &self.tm_num, &self.tm_den)
    }

    /// Set time mixing state.
    fn set_tm_state(&mut self, tm_last_x: Self::State, tm_num: Self::State, tm_den: Self::State) {
        self.tm_last_x = tm_last_x;
        self.tm_num = tm_num;
        self.tm_den = tm_den;
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
        (&x.into() * &self.0) + (&last_x.into() * (T::one() - &self.0))
    }
}

impl<T: ReqOps> RunAttention<T> for Attention<T, T> {
    type State = Array1<T>;
    fn time_mixing<S: HasRWKVLayerState<T, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State {
        let (last_x, last_num, last_den) = state.get_tm_state();

        let k = pardot(&self.key_weight, &self.time.mix_k.mix(&x, last_x));
        let v = pardot(&self.value_weight, &self.time.mix_v.mix(&x, last_x));
        let r = pardot(&self.receptance_weight, &self.time.mix_r.mix(&x, last_x));

        let exp_k = k.mapv(|el| el.exp());
        let exp_decay = self.time.decay.view();

        let wkv = {
            let e = (&*self.time.first + &k).mapv(|el| el.exp());
            (last_num + (&e * &v)) / (last_den + e)
        };
        let rwkv = r.sigmoid() * wkv;

        let num = (&exp_decay * last_num) + (&exp_k * &v);
        let den = (&exp_decay * last_den) + &exp_k;
        state.set_tm_state(x, num, den);
        pardot(&self.output_weight, &rwkv)
    }
}

impl<T: ReqOps> RunFFN<T> for FeedForwardNetwork<T, T> {
    type State = Array1<T>;
    fn channel_mixing<S: HasRWKVLayerState<T, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State {
        let last_x = state.get_cm_state();
        let k = pardot(&self.key_weight, &self.time.mix_k.mix(&x, last_x));
        let r = pardot(&self.receptance_weight, &self.time.mix_r.mix(&x, last_x));
        let vk = pardot(
            &self.value_weight,
            &k.mapv(|val| val.max(T::zero()).powi(2)),
        );

        state.set_cm_state(x);
        r.sigmoid() * &vk
    }
}

impl<T: ReqOps> RunLayerNorm for LayerNorm<T> {
    type XTy<'a> = ArrayView1<'a, T>;
    type Out = Array1<T>;
    fn norm<'a, X: Into<Self::XTy<'a>>>(&self, x: X) -> Self::Out {
        let origx = x.into();
        let x = &origx.view();
        let mean = x.mean().unwrap();
        let std = x.std(T::zero());
        (((x - mean) / std) * &self.weight) + &self.bias
    }
}

impl<T: ReqOps> RunRWKVLayer<T> for RWKVLayer<T, T> {
    type XTy = Array1<T>;
    type Out = Array1<T>;

    fn evaluate<S: HasRWKVLayerState<T, State = Self::Out>>(
        &self,
        x: Self::XTy,
        state: &mut S,
    ) -> Self::Out {
        let x = self.att.time_mixing(self.ln1.norm(&x), state) + &x;
        self.ffn.channel_mixing(self.ln2.norm(&x), state) + &x
    }
}

impl<T: ReqOps> RunRWKV<T> for RWKV<T, T> {
    type Out = Array1<T>;
    type Token = usize;

    fn evaluate<S: HasRWKVLayerState<T, State = Self::Out>>(
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

        let x = pardot(&self.head, &self.ln_out.norm(&x));
        let x_max = x.fold(T::min_value(), |acc, el| acc.max(*el));
        let e_x = (x - x_max).mapv(|el| el.exp());

        &e_x / e_x.sum()
    }
}
