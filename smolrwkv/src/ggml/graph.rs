#![allow(unused_imports, unused_variables, dead_code, unused_mut)]

use anyhow::{anyhow, Result};
use ndarray::ArrayView1;
use tokenizers::Tokenizer;

use ggml::{Context, Tensor, Type as GT};

use super::model::*;

mod map_ops {
    use super::{Context, Tensor};
    use std::{os::raw::c_int, slice};

    unsafe extern "C" fn one_minus_fun(n: c_int, dst: *mut f32, src: *mut f32) {
        let n = n as usize;
        let dst = slice::from_raw_parts_mut(dst, n);
        let src = slice::from_raw_parts(src, n);

        dst.iter_mut().zip(src.iter()).for_each(|(dstel, srcel)| {
            *dstel = 1.0 - *srcel;
        });
    }

    unsafe extern "C" fn sigmoid_fun(n: c_int, dst: *mut f32, src: *mut f32) {
        let n = n as usize;
        let dst = slice::from_raw_parts_mut(dst, n);
        let src = slice::from_raw_parts(src, n);

        dst.iter_mut()
            .zip(src.iter())
            .for_each(|(dstel, srcel)| *dstel = 1.0 / (1.0 + (-(*srcel)).exp()));
    }

    unsafe extern "C" fn relu_squared_fun(n: c_int, dst: *mut f32, src: *mut f32) {
        let n = n as usize;
        let dst = slice::from_raw_parts_mut(dst, n);
        let src = slice::from_raw_parts(src, n);

        dst.iter_mut()
            .zip(src.iter())
            .for_each(|(dstel, srcel)| *dstel = 0f32.max(*srcel).powi(2));
    }

    unsafe extern "C" fn max_fun(
        n: std::os::raw::c_int,
        dst: *mut f32,
        src0: *mut f32,
        src1: *mut f32,
    ) {
        let n = n as usize;
        let dst = std::slice::from_raw_parts_mut(dst, n);
        let src0 = std::slice::from_raw_parts(src0, n);
        let src1 = std::slice::from_raw_parts(src1, n);

        dst.iter_mut()
            .zip(src0.iter())
            .zip(src1.iter())
            .for_each(|((d, s0), s1)| {
                *d = s0.max(*s1);
            });
    }

    unsafe extern "C" fn sub_exp_fun(
        n: std::os::raw::c_int,
        dst: *mut f32,
        src0: *mut f32,
        src1: *mut f32,
    ) {
        let n = n as usize;
        let dst = std::slice::from_raw_parts_mut(dst, n);
        let src0 = std::slice::from_raw_parts(src0, n);
        let src1 = std::slice::from_raw_parts(src1, n);

        dst.iter_mut()
            .zip(src0.iter())
            .zip(src1.iter())
            .for_each(|((d, s0), s1)| {
                *d = (*s0 - *s1).exp();
            });
    }

    unsafe extern "C" fn div_fun(
        n: std::os::raw::c_int,
        dst: *mut f32,
        src0: *mut f32,
        src1: *mut f32,
    ) {
        let n = n as usize;
        let dst = std::slice::from_raw_parts_mut(dst, n);
        let src0 = std::slice::from_raw_parts(src0, n);
        let src1 = std::slice::from_raw_parts(src1, n);

        dst.iter_mut()
            .zip(src0.iter())
            .zip(src1.iter())
            .for_each(|((d, s0), s1)| {
                *d = *s0 / *s1;
            });
    }

    pub fn one_minus(ctx: &Context, tensor: &Tensor) -> Tensor {
        unsafe { ctx.op_map_unary(tensor, one_minus_fun) }
    }

    pub fn sigmoid(ctx: &Context, tensor: &Tensor) -> Tensor {
        unsafe { ctx.op_map_unary(tensor, sigmoid_fun) }
    }

    pub fn relu_squared(ctx: &Context, tensor: &Tensor) -> Tensor {
        unsafe { ctx.op_map_unary(tensor, relu_squared_fun) }
    }

    pub fn max(ctx: &Context, tensor1: &Tensor, tensor2: &Tensor) -> Tensor {
        unsafe { ctx.op_map_binary(tensor1, tensor2, max_fun) }
    }

    pub fn sub_exp(ctx: &Context, tensor1: &Tensor, tensor2: &Tensor) -> Tensor {
        unsafe { ctx.op_map_binary(tensor1, tensor2, sub_exp_fun) }
    }

    pub fn div(ctx: &Context, tensor1: &Tensor, tensor2: &Tensor) -> Tensor {
        unsafe { ctx.op_map_binary(tensor1, tensor2, div_fun) }
    }
}

impl LayerNorm {
    pub fn norm_ops(&self, ctx: &Context, x: &Tensor) -> Tensor {
        ctx.op_add(&ctx.op_mul(&ctx.op_norm(x), &self.weight), &self.bias)
    }
}

impl Mix {
    pub fn mix_ops(&self, ctx: &Context, x: &Tensor, last_x: &Tensor) -> Tensor {
        ctx.op_add(
            &ctx.op_mul(x, &self.0),
            &ctx.op_mul(last_x, &map_ops::one_minus(ctx, &self.0)),
        )
    }
}

impl FeedForwardNetwork {
    pub fn channel_mixing_ops(
        &self,
        ctx: &Context,
        state: &mut RWKVLayerState,
        x: Tensor,
    ) -> Tensor {
        let xk = &self.time.mix_k.mix_ops(ctx, &x, &state.cm_last_x);
        let xr = &self.time.mix_r.mix_ops(ctx, &x, &state.cm_last_x);

        let r = &map_ops::sigmoid(ctx, &ctx.op_mul_mat(&self.receptance_weight, xr));
        let k = &map_ops::relu_squared(ctx, &ctx.op_mul_mat(&self.key_weight, xk));

        state.cm_last_x = x;
        ctx.op_mul(r, &ctx.op_mul_mat(&self.value_weight, k))
    }
}

impl Attention {
    pub fn time_mixing_ops(&self, ctx: &Context, state: &mut RWKVLayerState, x: Tensor) -> Tensor {
        let (tm_last_x, aa, bb, pp) = (&state.tm_last_x, &state.tm_aa, &state.tm_bb, &state.tm_pp);

        let xk = &self.time.mix_k.mix_ops(ctx, &x, tm_last_x);
        let xv = &self.time.mix_v.mix_ops(ctx, &x, tm_last_x);
        let xr = &self.time.mix_r.mix_ops(ctx, &x, tm_last_x);

        let r = &map_ops::sigmoid(ctx, &ctx.op_mul_mat(&self.receptance_weight, xr));
        let k = &ctx.op_mul_mat(&self.key_weight, xk);
        let v = &ctx.op_mul_mat(&self.value_weight, xv);

        let (a, b) = {
            let ww = &ctx.op_add(&self.time.first, k);
            let qq = &map_ops::max(ctx, ww, pp);
            let e1 = &map_ops::sub_exp(ctx, pp, qq);
            let e2 = &map_ops::sub_exp(ctx, ww, qq);
            let a = ctx.op_add(&ctx.op_mul(e1, aa), &ctx.op_mul(e2, v));
            let b = ctx.op_add(&ctx.op_mul(e1, bb), e2);
            (a, b)
        };

        let (wkv, new_aa, new_bb, new_pp) = {
            let ww = &ctx.op_add(pp, &self.time.decay);
            let qq = map_ops::max(ctx, ww, k);
            let e1 = &map_ops::sub_exp(ctx, ww, &qq);
            let e2 = &map_ops::sub_exp(ctx, k, &qq);
            let wkv = map_ops::div(ctx, &a, &b);

            let new_aa = ctx.op_add(&ctx.op_mul(e1, aa), &ctx.op_mul(e2, v));
            let new_bb = ctx.op_add(&ctx.op_mul(e1, bb), e2);
            let new_pp = qq;

            (wkv, new_aa, new_bb, new_pp)
        };

        state.tm_last_x = x;
        state.tm_aa = new_aa;
        state.tm_bb = new_bb;
        state.tm_pp = new_pp;

        ctx.op_mul_mat(&self.output_weight, &ctx.op_mul(r, &wkv))
    }
}

impl RWKVLayer {
    pub fn evaluate_layer_ops(
        &self,
        ctx: &Context,
        state: &mut RWKVLayerState,
        x: Tensor,
    ) -> Tensor {
        let x = ctx.op_add(
            &self
                .att
                .time_mixing_ops(ctx, state, self.ln_tm.norm_ops(ctx, &x)),
            &x,
        );
        ctx.op_add(
            &self
                .ffn
                .channel_mixing_ops(ctx, state, self.ln_cm.norm_ops(ctx, &x)),
            &x,
        )
    }
}

impl RWKV {
    pub fn evaluate_ops(
        &self,
        ctx: &Context,
        state: &mut [RWKVLayerState],
        token: usize,
    ) -> Tensor {
        // FIXME: Get initial state.
        let initial_x = ctx.new_f32(666.666);

        let x = self
            .layers
            .iter()
            .enumerate()
            .fold(initial_x, |x, (lnum, layer)| {
                layer.evaluate_layer_ops(ctx, &mut state[lnum], x)
            });
        ctx.op_mul_mat(&self.head_weight, &self.ln_out.norm_ops(ctx, &x))
    }
}
