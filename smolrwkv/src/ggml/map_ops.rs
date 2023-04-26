use rusty_ggml::{dims::*, prelude::GTensor};

#[cfg(all(feature = "simd", feature = "never_happening"))]
// I don't notice a performance difference. :(
mod ops_funs {
    use num_traits::FromPrimitive;
    use simba::simd::{SimdComplexField, SimdPartialOrd, WideF32x8};
    use std::{os::raw::c_int, slice};

    const SIMD_CHUNKSIZE: usize = 8;
    type SimdTyp = WideF32x8;

    pub unsafe extern "C" fn one_minus_fun(n: c_int, dst: *mut f32, src: *const f32) {
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

    pub unsafe extern "C" fn sigmoid_fun(n: c_int, dst: *mut f32, src: *const f32) {
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

    pub unsafe extern "C" fn relu_squared_fun(n: c_int, dst: *mut f32, src: *const f32) {
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

    pub unsafe extern "C" fn max_fun(n: c_int, dst: *mut f32, src0: *const f32, src1: *const f32) {
        let n = n as usize;
        let dst = slice::from_raw_parts_mut(dst, n);
        let src0 = slice::from_raw_parts(src0, n);
        let src1 = slice::from_raw_parts(src1, n);

        let mut itd = dst.chunks_exact_mut(SIMD_CHUNKSIZE);
        let mut its0 = src0.chunks_exact(SIMD_CHUNKSIZE);
        let mut its1 = src1.chunks_exact(SIMD_CHUNKSIZE);

        (&mut itd)
            .zip(&mut its0)
            .zip(&mut its1)
            .for_each(|((dstchunk, src0chunk), src1chunk)| {
                let src0chunk: [f32; SIMD_CHUNKSIZE] =
                    unsafe { src0chunk.try_into().unwrap_unchecked() };
                let src1chunk: [f32; SIMD_CHUNKSIZE] =
                    unsafe { src1chunk.try_into().unwrap_unchecked() };
                dstchunk.copy_from_slice(
                    &(SimdTyp::from(src0chunk).simd_max(SimdTyp::from(src1chunk)))
                        .0
                        .to_array(),
                );
            });
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
        src0: *const f32,
        src1: *const f32,
    ) {
        let n = n as usize;
        let dst = slice::from_raw_parts_mut(dst, n);
        let src0 = slice::from_raw_parts(src0, n);
        let src1 = slice::from_raw_parts(src1, n);

        let mut itd = dst.chunks_exact_mut(SIMD_CHUNKSIZE);
        let mut its0 = src0.chunks_exact(SIMD_CHUNKSIZE);
        let mut its1 = src1.chunks_exact(SIMD_CHUNKSIZE);

        (&mut itd)
            .zip(&mut its0)
            .zip(&mut its1)
            .for_each(|((dstchunk, src0chunk), src1chunk)| {
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
            });
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

    pub unsafe extern "C" fn div_fun(n: c_int, dst: *mut f32, src0: *const f32, src1: *const f32) {
        let n = n as usize;
        let dst = slice::from_raw_parts_mut(dst, n);
        let src0 = slice::from_raw_parts(src0, n);
        let src1 = slice::from_raw_parts(src1, n);

        let mut itd = dst.chunks_exact_mut(SIMD_CHUNKSIZE);
        let mut its0 = src0.chunks_exact(SIMD_CHUNKSIZE);
        let mut its1 = src1.chunks_exact(SIMD_CHUNKSIZE);

        (&mut itd)
            .zip(&mut its0)
            .zip(&mut its1)
            .for_each(|((dstchunk, src0chunk), src1chunk)| {
                let src0chunk: [f32; SIMD_CHUNKSIZE] =
                    unsafe { src0chunk.try_into().unwrap_unchecked() };
                let src1chunk: [f32; SIMD_CHUNKSIZE] =
                    unsafe { src1chunk.try_into().unwrap_unchecked() };
                dstchunk.copy_from_slice(
                    &(SimdTyp::from(src0chunk) / SimdTyp::from(src1chunk))
                        .0
                        .to_array(),
                );
            });
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

    pub unsafe extern "C" fn one_minus_fun(n: c_int, dst: *mut f32, src: *const f32) {
        let n = n as usize;
        let dst = slice::from_raw_parts_mut(dst, n);
        let src = slice::from_raw_parts(src, n);

        dst.iter_mut()
            .zip(src.iter())
            .for_each(|(dstel, srcel)| *dstel = 1.0 - *srcel);
    }

    pub unsafe extern "C" fn sigmoid_fun(n: c_int, dst: *mut f32, src: *const f32) {
        let n = n as usize;
        let dst = slice::from_raw_parts_mut(dst, n);
        let src = slice::from_raw_parts(src, n);

        dst.iter_mut()
            .zip(src.iter())
            .for_each(|(dstel, srcel)| *dstel = 1.0 / (1.0 + (-(*srcel)).exp()));
    }

    pub unsafe extern "C" fn relu_squared_fun(n: c_int, dst: *mut f32, src: *const f32) {
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
        src0: *const f32,
        src1: *const f32,
    ) {
        let n = n as usize;
        let dst = std::slice::from_raw_parts_mut(dst, n);
        let src0 = std::slice::from_raw_parts(src0, n);
        let src1 = std::slice::from_raw_parts(src1, n);

        dst.iter_mut()
            .zip(src0.iter())
            .zip(src1.iter())
            .for_each(|((dstel, src0el), src1el)| {
                *dstel = src0el.max(*src1el);
            });
    }

    pub unsafe extern "C" fn sub_exp_fun(
        n: std::os::raw::c_int,
        dst: *mut f32,
        src0: *const f32,
        src1: *const f32,
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

    pub unsafe extern "C" fn scalar_div_fun(
        n: std::os::raw::c_int,
        dst: *mut f32,
        src0: *const f32,
        src1: *const f32,
    ) {
        let n = n as usize;
        let dst = std::slice::from_raw_parts_mut(dst, n);
        let src0 = std::slice::from_raw_parts(src0, n);
        let src1 = std::slice::from_raw_parts(src1, n);

        dst.iter_mut()
            .zip(src0.iter())
            .zip(src1.iter())
            .for_each(|((dstel, src0el), src1el)| {
                *dstel = *src0el / *src1el;
            });
    }
}

use ops_funs::*;

pub fn one_minus<const DIMS: usize, T: AsRef<GTensor<DIMS>>>(tensor: T) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor.as_ref().map_unary(one_minus_fun)
}

pub fn sigmoid<const DIMS: usize, T: AsRef<GTensor<DIMS>>>(tensor: T) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor.as_ref().map_unary(sigmoid_fun)
}

pub fn relu_squared<const DIMS: usize, T: AsRef<GTensor<DIMS>>>(tensor: T) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor.as_ref().map_unary(relu_squared_fun)
}

pub fn max<const DIMS: usize, T1: AsRef<GTensor<DIMS>>, T2: AsRef<GTensor<DIMS>>>(
    tensor1: T1,
    tensor2: T2,
) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor1.as_ref().map_binary(tensor2, max_fun)
}

pub fn sub_exp<const DIMS: usize, T1: AsRef<GTensor<DIMS>>, T2: AsRef<GTensor<DIMS>>>(
    tensor1: T1,
    tensor2: T2,
) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor1.as_ref().map_binary(tensor2, sub_exp_fun)
}

pub fn div_scalar<const DIMS: usize, T1: AsRef<GTensor<DIMS>>, T2: AsRef<GTensor<DIMS>>>(
    tensor1: T1,
    tensor2: T2,
) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor1.as_ref().map_binary(tensor2, scalar_div_fun)
}
