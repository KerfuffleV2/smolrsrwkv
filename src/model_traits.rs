#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array1, AsArray, Ix1};

use crate::model::RWKVLayerState;

pub trait HasRWKVLayerState<T> {
    type State;
    fn get_tm_state(&self) -> (&Self::State, &Self::State, &Self::State);
    fn set_tm_state(&mut self, tm_last_x: Self::State, tm_num: Self::State, tm_den: Self::State);
    fn get_cm_state(&self) -> &Self::State;
    fn set_cm_state(&mut self, cm_last_x: Self::State);
}

pub trait RunMix<T> {
    type Out;
    fn mix<'a, A: AsArray<'a, T, Ix1>>(&self, x: A, last_x: A) -> Self::Out
    where
        T: 'a;
}

pub trait RunAttention<T> {
    type State;
    fn time_mixing<S: HasRWKVLayerState<T, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State;
}

pub trait RunFFN<T> {
    type State;
    fn channel_mixing<S: HasRWKVLayerState<T, State = Self::State>>(
        &self,
        x: Self::State,
        state: &mut S,
    ) -> Self::State;
}

pub trait RunLayerNorm<T> {
    type Out;
    /// Normalize a 1D array.
    fn norm<'a, A: AsArray<'a, T, Ix1>>(&self, x: A) -> Self::Out
    where
        T: 'a;
}

pub trait RunRWKVLayer<T> {
    type Out;

    /// Evaluates a layer. Each layer must be evaluated in sequence,
    /// serially as they each generate "x" and also require "x" as input.
    fn evaluate(&self, x: Array1<T>, state: &mut RWKVLayerState<T>) -> Self::Out;
}

pub trait RunRWKV<T> {
    type Out;
    fn evaluate(&self, token: usize, state: &mut [RWKVLayerState<T>]) -> Self::Out;
}
