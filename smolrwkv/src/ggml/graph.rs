use ggml::{Context, Tensor};

use super::model::*;

#[allow(dead_code, unused_imports)]
mod map_ops {
    use super::{Context, Tensor};

    #[cfg(feature = "simd")]
    // I don't notice a performance difference. :(
    mod ops_funs {
        use num_traits::FromPrimitive;
        use simba::simd::{SimdComplexField, SimdPartialOrd, WideF32x8};
        use std::{os::raw::c_int, slice};

        const SIMD_CHUNKSIZE: usize = 8;
        type SimdTyp = WideF32x8;

        pub unsafe extern "C" fn one_minus_fun(n: c_int, dst: *mut f32, src: *mut f32) {
            let n = n as usize;
            let dst = slice::from_raw_parts_mut(dst, n);
            let src = slice::from_raw_parts(src, n);

            let mut itd = dst.chunks_exact_mut(SIMD_CHUNKSIZE);
            let mut its = src.chunks_exact(SIMD_CHUNKSIZE);

            let wv = unsafe { SimdTyp::from_f32(1.0).unwrap_unchecked() };
            (&mut itd).zip(&mut its).for_each(|(dstchunk, srcchunk)| {
                let src0chunk: [f32; SIMD_CHUNKSIZE] =
                    unsafe { srcchunk.try_into().unwrap_unchecked() };
                dstchunk.copy_from_slice(&(wv - SimdTyp::from(src0chunk)).0.to_array());
            });
            let (rems, remd) = (its.remainder(), itd.into_remainder());
            if !rems.is_empty() {
                remd.iter_mut().zip(rems.iter()).for_each(|(dstel, srcel)| {
                    *dstel = 1.0 - *srcel;
                });
            }
        }

        pub unsafe extern "C" fn sigmoid_fun(n: c_int, dst: *mut f32, src: *mut f32) {
            let n = n as usize;
            let dst = slice::from_raw_parts_mut(dst, n);
            let src = slice::from_raw_parts(src, n);

            let mut itd = dst.chunks_exact_mut(SIMD_CHUNKSIZE);
            let mut its = src.chunks_exact(SIMD_CHUNKSIZE);

            let wv = unsafe { SimdTyp::from_f32(1.0).unwrap_unchecked() };
            (&mut itd).zip(&mut its).for_each(|(dstchunk, srcchunk)| {
                let src0chunk: [f32; SIMD_CHUNKSIZE] =
                    unsafe { srcchunk.try_into().unwrap_unchecked() };
                dstchunk.copy_from_slice(
                    &(wv / (wv + (-SimdTyp::from(src0chunk)).simd_exp()))
                        .0
                        .to_array(),
                );
            });
            let (rems, remd) = (its.remainder(), itd.into_remainder());
            if !rems.is_empty() {
                remd.iter_mut().zip(rems.iter()).for_each(|(dstel, srcel)| {
                    *dstel = 1.0 / (1.0 + (-(*srcel)).exp());
                });
            }
        }

        pub unsafe extern "C" fn relu_squared_fun(n: c_int, dst: *mut f32, src: *mut f32) {
            let n = n as usize;
            let dst = slice::from_raw_parts_mut(dst, n);
            let src = slice::from_raw_parts(src, n);

            let mut itd = dst.chunks_exact_mut(SIMD_CHUNKSIZE);
            let mut its = src.chunks_exact(SIMD_CHUNKSIZE);

            let zv = unsafe { SimdTyp::from_f32(0.0).unwrap_unchecked() };
            (&mut itd).zip(&mut its).for_each(|(dstchunk, srcchunk)| {
                let src0chunk: [f32; SIMD_CHUNKSIZE] =
                    unsafe { srcchunk.try_into().unwrap_unchecked() };
                dstchunk.copy_from_slice(
                    &(zv.simd_max(SimdTyp::from(src0chunk)).simd_powi(2))
                        .0
                        .to_array(),
                );
            });
            let (rems, remd) = (its.remainder(), itd.into_remainder());
            if !rems.is_empty() {
                remd.iter_mut().zip(rems.iter()).for_each(|(dstel, srcel)| {
                    *dstel = 0f32.max(*srcel).powi(2);
                });
            }
        }

        pub unsafe extern "C" fn max_fun(n: c_int, dst: *mut f32, src0: *mut f32, src1: *mut f32) {
            let n = n as usize;
            let dst = slice::from_raw_parts_mut(dst, n);
            let src0 = slice::from_raw_parts(src0, n);
            let src1 = slice::from_raw_parts(src1, n);

            let mut itd = dst.chunks_exact_mut(SIMD_CHUNKSIZE);
            let mut its0 = src0.chunks_exact(SIMD_CHUNKSIZE);
            let mut its1 = src1.chunks_exact(SIMD_CHUNKSIZE);

            (&mut itd).zip(&mut its0).zip(&mut its1).for_each(
                |((dstchunk, src0chunk), src1chunk)| {
                    let src0chunk: [f32; SIMD_CHUNKSIZE] =
                        unsafe { src0chunk.try_into().unwrap_unchecked() };
                    let src1chunk: [f32; SIMD_CHUNKSIZE] =
                        unsafe { src1chunk.try_into().unwrap_unchecked() };
                    dstchunk.copy_from_slice(
                        &(SimdTyp::from(src0chunk).simd_max(SimdTyp::from(src1chunk)))
                            .0
                            .to_array(),
                    );
                },
            );
            let (rems0, rems1, remd) = (its0.remainder(), its1.remainder(), itd.into_remainder());
            if !rems0.is_empty() {
                remd.iter_mut()
                    .zip(rems0.iter())
                    .zip(rems1.iter())
                    .for_each(|((dstel, src0el), src1el)| {
                        *dstel = src0el.max(*src1el);
                    });
            }
        }

        pub unsafe extern "C" fn sub_exp_fun(
            n: c_int,
            dst: *mut f32,
            src0: *mut f32,
            src1: *mut f32,
        ) {
            let n = n as usize;
            let dst = slice::from_raw_parts_mut(dst, n);
            let src0 = slice::from_raw_parts(src0, n);
            let src1 = slice::from_raw_parts(src1, n);

            let mut itd = dst.chunks_exact_mut(SIMD_CHUNKSIZE);
            let mut its0 = src0.chunks_exact(SIMD_CHUNKSIZE);
            let mut its1 = src1.chunks_exact(SIMD_CHUNKSIZE);

            (&mut itd).zip(&mut its0).zip(&mut its1).for_each(
                |((dstchunk, src0chunk), src1chunk)| {
                    let src0chunk: [f32; SIMD_CHUNKSIZE] =
                        unsafe { src0chunk.try_into().unwrap_unchecked() };
                    let src1chunk: [f32; SIMD_CHUNKSIZE] =
                        unsafe { src1chunk.try_into().unwrap_unchecked() };
                    dstchunk.copy_from_slice(
                        &(SimdTyp::from(src0chunk) - SimdTyp::from(src1chunk))
                            .simd_exp()
                            .0
                            .to_array(),
                    );
                },
            );
            let (rems0, rems1, remd) = (its0.remainder(), its1.remainder(), itd.into_remainder());
            if !rems0.is_empty() {
                remd.iter_mut()
                    .zip(rems0.iter())
                    .zip(rems1.iter())
                    .for_each(|((dstel, src0el), src1el)| {
                        *dstel = (*src0el - *src1el).exp();
                    });
            }
        }

        pub unsafe extern "C" fn div_fun(n: c_int, dst: *mut f32, src0: *mut f32, src1: *mut f32) {
            let n = n as usize;
            let dst = slice::from_raw_parts_mut(dst, n);
            let src0 = slice::from_raw_parts(src0, n);
            let src1 = slice::from_raw_parts(src1, n);

            let mut itd = dst.chunks_exact_mut(SIMD_CHUNKSIZE);
            let mut its0 = src0.chunks_exact(SIMD_CHUNKSIZE);
            let mut its1 = src1.chunks_exact(SIMD_CHUNKSIZE);

            (&mut itd).zip(&mut its0).zip(&mut its1).for_each(
                |((dstchunk, src0chunk), src1chunk)| {
                    let src0chunk: [f32; SIMD_CHUNKSIZE] =
                        unsafe { src0chunk.try_into().unwrap_unchecked() };
                    let src1chunk: [f32; SIMD_CHUNKSIZE] =
                        unsafe { src1chunk.try_into().unwrap_unchecked() };
                    dstchunk.copy_from_slice(
                        &(SimdTyp::from(src0chunk) / SimdTyp::from(src1chunk))
                            .0
                            .to_array(),
                    );
                },
            );
            let (rems0, rems1, remd) = (its0.remainder(), its1.remainder(), itd.into_remainder());
            if !rems0.is_empty() {
                remd.iter_mut()
                    .zip(rems0.iter())
                    .zip(rems1.iter())
                    .for_each(|((dstel, src0el), src1el)| {
                        *dstel = *src0el / *src1el;
                    });
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    mod ops_funs {
        use std::{os::raw::c_int, slice};

        pub unsafe extern "C" fn one_minus_fun(n: c_int, dst: *mut f32, src: *mut f32) {
            let n = n as usize;
            let dst = slice::from_raw_parts_mut(dst, n);
            let src = slice::from_raw_parts(src, n);

            dst.iter_mut()
                .zip(src.iter())
                .for_each(|(dstel, srcel)| *dstel = 1.0 - *srcel);
        }

        pub unsafe extern "C" fn sigmoid_fun(n: c_int, dst: *mut f32, src: *mut f32) {
            let n = n as usize;
            let dst = slice::from_raw_parts_mut(dst, n);
            let src = slice::from_raw_parts(src, n);

            dst.iter_mut()
                .zip(src.iter())
                .for_each(|(dstel, srcel)| *dstel = 1.0 / (1.0 + (-(*srcel)).exp()));
        }

        pub unsafe extern "C" fn relu_squared_fun(n: c_int, dst: *mut f32, src: *mut f32) {
            let n = n as usize;
            let dst = slice::from_raw_parts_mut(dst, n);
            let src = slice::from_raw_parts(src, n);

            dst.iter_mut()
                .zip(src.iter())
                .for_each(|(dstel, srcel)| *dstel = 0f32.max(*srcel).powi(2));
        }

        pub unsafe extern "C" fn max_fun(
            n: std::os::raw::c_int,
            dst: *mut f32,
            src0: *mut f32,
            src1: *mut f32,
        ) {
            let n = n as usize;
            let dst = std::slice::from_raw_parts_mut(dst, n);
            let src0 = std::slice::from_raw_parts(src0, n);
            let src1 = std::slice::from_raw_parts(src1, n);

            dst.iter_mut().zip(src0.iter()).zip(src1.iter()).for_each(
                |((dstel, src0el), src1el)| {
                    *dstel = src0el.max(*src1el);
                },
            );
        }

        pub unsafe extern "C" fn sub_exp_fun(
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

        pub unsafe extern "C" fn div_fun(
            n: std::os::raw::c_int,
            dst: *mut f32,
            src0: *mut f32,
            src1: *mut f32,
        ) {
            let n = n as usize;
            let dst = std::slice::from_raw_parts_mut(dst, n);
            let src0 = std::slice::from_raw_parts(src0, n);
            let src1 = std::slice::from_raw_parts(src1, n);

            dst.iter_mut().zip(src0.iter()).zip(src1.iter()).for_each(
                |((dstel, src0el), src1el)| {
                    *dstel = *src0el / *src1el;
                },
            );
        }
    }

    use ops_funs::*;

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

        state.cm_last_x = ctx.op_cpy(&x, &state.cm_last_x);
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

        state.tm_last_x = ctx.op_cpy(&x, &state.tm_last_x);
        state.tm_aa = ctx.op_cpy(&new_aa, &state.tm_aa);
        state.tm_bb = ctx.op_cpy(&new_bb, &state.tm_bb);
        state.tm_pp = ctx.op_cpy(&new_pp, &state.tm_pp);

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
        token: Tensor,
    ) -> Tensor {
        let initial_x = ctx.op_get_rows(&self.emb, &token);
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
