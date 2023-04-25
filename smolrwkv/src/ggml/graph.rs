use rusty_ggml::tensor::GgmlTensor as Tensor;

use super::{map_ops, model::*};

impl LayerNorm {
    pub fn norm_ops<T: AsRef<Tensor<1>>>(&self, x: T) -> Tensor<1> {
        (x.as_ref().norm() * &self.weight) + &self.bias
    }
}

impl Mix {
    pub fn mix_ops<TX: AsRef<Tensor<1>>, TLX: AsRef<Tensor<1>>>(
        &self,
        x: TX,
        last_x: TLX,
    ) -> Tensor<1> {
        (x.as_ref() * &self.0) + (last_x.as_ref() * map_ops::one_minus(&self.0))
    }
}

impl FeedForwardNetwork {
    pub fn channel_mixing_ops(&self, state: &mut RWKVLayerState, x: Tensor<1>) -> Tensor<1> {
        let xk = self.time.mix_k.mix_ops(&x, &state.cm_last_x);
        let xr = self.time.mix_r.mix_ops(&x, &state.cm_last_x);

        let r = map_ops::sigmoid(self.receptance_weight.mul_mat_smallrhs(xr));
        let k = &map_ops::relu_squared(self.key_weight.mul_mat_smallrhs(xk));

        state.cm_last_x = state.cm_last_x.clone().copy_from(x);
        r * self.value_weight.mul_mat_smallrhs(k)
    }
}

impl Attention {
    pub fn time_mixing_ops(&self, state: &mut RWKVLayerState, x: Tensor<1>) -> Tensor<1> {
        let (tm_last_x, aa, bb, pp) = (&state.tm_last_x, &state.tm_aa, &state.tm_bb, &state.tm_pp);

        let xk = self.time.mix_k.mix_ops(&x, tm_last_x);
        let xv = self.time.mix_v.mix_ops(&x, tm_last_x);
        let xr = self.time.mix_r.mix_ops(&x, tm_last_x);

        let r = map_ops::sigmoid(self.receptance_weight.mul_mat_smallrhs(xr));
        let k = &self.key_weight.mul_mat_smallrhs(xk);
        let v = &self.value_weight.mul_mat_smallrhs(xv);

        let (a, b) = {
            let ww = &self.time.first + k;
            let qq = map_ops::max(&ww, pp);
            let e1 = map_ops::sub_exp(pp, &qq);
            let e2 = map_ops::sub_exp(ww, qq);
            let a = &e1 * aa + &e2 * v;
            let b = (e1 * bb) + e2;
            (a, b)
        };

        let (wkv, new_aa, new_bb, new_pp) = {
            let ww = pp + &self.time.decay;
            let qq = map_ops::max(&ww, k);
            let e1 = map_ops::sub_exp(ww, &qq);
            let e2 = map_ops::sub_exp(k, &qq);
            let wkv = a / b;

            let new_aa = &e1 * aa + &e2 * v;
            let new_bb = (e1 * bb) + e2;
            let new_pp = qq;

            (wkv, new_aa, new_bb, new_pp)
        };

        state.tm_last_x = state.tm_last_x.clone().copy_from(x);
        state.tm_aa = state.tm_aa.clone().copy_from(new_aa);
        state.tm_bb = state.tm_bb.clone().copy_from(new_bb);
        state.tm_pp = state.tm_pp.clone().copy_from(new_pp);

        self.output_weight.mul_mat_smallrhs(r * wkv)
    }
}

impl RWKVLayer {
    pub fn evaluate_layer_ops(&self, state: &mut RWKVLayerState, x: Tensor<1>) -> Tensor<1> {
        let x = self.att.time_mixing_ops(state, self.ln_tm.norm_ops(&x)) + x;
        self.ffn.channel_mixing_ops(state, self.ln_cm.norm_ops(&x)) + x
    }
}

impl RWKV {
    pub fn evaluate_ops(&self, state: &mut [RWKVLayerState], token: Tensor<1>) -> Tensor<1> {
        let initial_x = self.emb.get_rows(token);
        let initial_x = initial_x.view([initial_x.elements() as i64], [0]);
        let x = self
            .layers
            .iter()
            .enumerate()
            .fold(initial_x, |x, (lnum, layer)| {
                layer.evaluate_layer_ops(&mut state[lnum], x)
            });
        self.head_weight.mul_mat_smallrhs(self.ln_out.norm_ops(&x))
    }
}
