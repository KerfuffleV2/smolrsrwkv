#![allow(dead_code, unused_imports, unused_variables, non_snake_case)]
use std::{marker::PhantomData, ops::Sub};

use crate::util::ReqOps;
use anyhow::{anyhow, Result};
use mmap_rs::{MmapFlags, MmapOptions};
use ndarray::{
    Array, Array1, Array2, ArrayView1, ArrayView2, AsArray, DimMax, Dimension, Ix1, Ix2, NdFloat,
    ScalarOperand, Zip,
};
use num_traits::{FromPrimitive, One, Zero};

pub trait TyEq {}

impl<T> TyEq for (T, T) {}

#[derive(Clone, Copy, Default)]
pub struct RWKVOps<T>(PhantomData<*const T>);
pub const ROF32: RWKVOps<f32> = RWKVOps(PhantomData);

impl RWKVOps<f32> {
    pub fn sigmoid<'a, A: AsArray<'a, f32, Ix1>>(&self, x: A) -> Array1<f32> {
        let x = x.into();
        x.map(|val| 1.0 / (1.0 + (-(*val)).exp()))
    }
}

pub trait RWKVOps11<'a, T> {
    type Out;
    fn sigmoid(self) -> Self::Out;
}

struct Merp(Array1<f32>);
impl<'a> RWKVOps11<'a, f32> for &'a Merp {
    type Out = Merp;
    fn sigmoid(self) -> Merp {
        Merp(self.0.sigmoid())
    }
}

impl<'a, T: ReqOps, A: AsArray<'a, T, Ix1>> RWKVOps11<'a, T> for A {
    type Out = Array1<T>;
    fn sigmoid(self) -> Self::Out {
        self.into()
            .map(|val| T::one() / (T::one() + (-(*val)).exp()))
    }
}

fn durp() {
    let x: Array1<f32> = Array1::zeros(100);
    let y = ROF32.sigmoid(&x);
}

// pub trait RWKVOpsD1D1 {
//     type Element;
//     type Result<U>;

//     fn sigmoid<U>(&self) -> Self::Result<U>;
// }

// impl RWKVOpsD1D1 for Array1<f32> {
//     type Element = f32;
//     type Result<U> = Array1<U>;

//     fn sigmoid<U>(&self) -> Self::Result<U> {
//         let x = self.map(|v| Self::Element::one());
//         unsafe { std::mem::transmute(x) }
//         // self.map(|val| Self::Element::one() / (Self::Element::one() + (-(*val)).exp()))
//         // todo!()
//     }
// }
