#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array1, AsArray, Axis, Ix1};

use crate::model::*;
use crate::model_traits::*;
use crate::rwkvops::RWKVOps11;
use crate::util::{pardot, ReqOps};

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

impl<T: ReqOps> RunMix<T> for Mix<T> {
    type Out = Array1<T>;

    fn mix<'a, A: AsArray<'a, T, Ix1>>(&self, x: A, last_x: A) -> Self::Out
    where
        T: 'a,
    {
        (&x.into() * &self.0) + (&last_x.into() * (T::one() - &self.0))
    }
}

impl<T: ReqOps> RunAttention<T> for Attention<T> {
    type State = Array1<T>;
    fn time_mixing<S: HasRWKVLayerState<T, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State {
        let xv = &x.view();
        let (last_x, last_num, last_den) = state.get_tm_state();
        let last_x = &last_x.view();

        let k = pardot(&self.key_weight, &self.time.mix_k.mix(xv, last_x));
        let v = pardot(&self.value_weight, &self.time.mix_v.mix(xv, last_x));
        let r = pardot(&self.receptance_weight, &self.time.mix_r.mix(xv, last_x));

        let exp_k = k.mapv(|el| el.exp());
        let exp_decay = self.time.decay.mapv(|el| (-el.exp()).exp());

        let wkv = {
            let e = (&self.time.first + &k).mapv(|el| el.exp());
            (last_num + (&e * &v)) / (last_den + e)
        };
        let rwkv = r.sigmoid() * wkv;

        let num = (&exp_decay * last_num) + (&exp_k * &v);
        let den = (&exp_decay * last_den) + &exp_k;
        state.set_tm_state(x, num, den);
        pardot(&self.output_weight, &rwkv)
    }
}

impl<T: ReqOps> RunFFN<T> for FeedForwardNetwork<T> {
    type State = Array1<T>;
    fn channel_mixing<S: HasRWKVLayerState<T, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State {
        let xv = &x.view();
        let last_x = &state.get_cm_state().view();
        let k = pardot(&self.key_weight, &self.time.mix_k.mix(xv, last_x));
        let r = pardot(&self.receptance_weight, &self.time.mix_r.mix(xv, last_x));
        let vk = pardot(
            &self.value_weight,
            &k.mapv(|val| val.max(T::zero()).powi(2)),
        );

        state.set_cm_state(x);
        r.sigmoid() * &vk
    }
}

impl<T: ReqOps> RunLayerNorm<T> for LayerNorm<T> {
    type Out = Array1<T>;
    fn norm<'a, A: AsArray<'a, T, Ix1>>(&self, x: A) -> Self::Out
    where
        T: 'a,
    {
        let origx = x.into();
        let x = &origx.view();
        let mean = x.mean().unwrap();
        let std = x.std(T::zero());
        (((x - mean) / std) * &self.weight) + &self.bias
    }
}

impl<T: ReqOps> RunRWKVLayer<T> for RWKVLayer<T> {
    type Out = Array1<T>;

    fn evaluate(&self, x: Array1<T>, state: &mut RWKVLayerState<T>) -> Self::Out {
        let x = self.att.time_mixing(self.ln1.norm(&x), state) + &x;
        self.ffn.channel_mixing(self.ln2.norm(&x), state) + &x
    }
}

impl<T: ReqOps> RunRWKV<T> for RWKV<T> {
    type Out = Array1<T>;

    fn evaluate(&self, token: usize, state: &mut [RWKVLayerState<T>]) -> Self::Out {
        let initial_x = self.ln0.norm(self.emb.index_axis(Axis(0), token));

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
