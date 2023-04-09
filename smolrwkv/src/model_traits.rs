#![allow(clippy::upper_case_acronyms)]

pub trait HasRWKVLayerState<T> {
    type State;
    fn get_tm_state(&self) -> (&Self::State, &Self::State, &Self::State, &Self::State);
    fn set_tm_state(
        &mut self,
        tm_last_x: Self::State,
        aa: Self::State,
        bb: Self::State,
        pp: Self::State,
    );
    fn get_cm_state(&self) -> &Self::State;
    fn set_cm_state(&mut self, cm_last_x: Self::State);
}

pub trait RunMix {
    type XTy<'a>;
    type Out;
    fn mix<'a, X: Into<Self::XTy<'a>>, LX: Into<Self::XTy<'a>>>(
        &self,
        x: X,
        last_x: LX,
    ) -> Self::Out;
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

pub trait RunLayerNorm {
    type XTy<'a>;
    type Out;
    /// Normalize a 1D array.
    fn norm<'a, X: Into<Self::XTy<'a>>>(&self, x: X) -> Self::Out;
}

pub trait RunRWKVLayer<T> {
    type XTy;
    type Out;

    /// Evaluates a layer. Each layer must be evaluated in sequence,
    /// serially as they each generate "x" and also require "x" as input.
    fn evaluate<S: HasRWKVLayerState<T, State = Self::Out>>(
        &self,
        x: Self::XTy,
        state: &mut S,
    ) -> Self::Out;
}

pub trait RunRWKV<T> {
    type Out;
    type Token;
    fn evaluate<S: HasRWKVLayerState<T, State = Self::Out>>(
        &self,
        token: Self::Token,
        state: &mut [S],
    ) -> Self::Out;
}
