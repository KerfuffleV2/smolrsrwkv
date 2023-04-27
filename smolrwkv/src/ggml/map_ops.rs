use rusty_ggml::{
    dims::*,
    prelude::{map_binop, map_unop, GTensor},
};

pub fn one_minus<const DIMS: usize, T: AsRef<GTensor<DIMS>>>(tensor: T) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor.as_ref().map_unary(map_unop!(|el| 1.0 - el))
}

pub fn sigmoid<const DIMS: usize, T: AsRef<GTensor<DIMS>>>(tensor: T) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor
        .as_ref()
        .map_unary(map_unop!(|el| 1.0 / (1.0 + (-(el)).exp())))
}

pub fn relu_squared<const DIMS: usize, T: AsRef<GTensor<DIMS>>>(tensor: T) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor
        .as_ref()
        .map_unary(map_unop!(|el| 0f32.max(el).powi(2)))
}

pub fn max<const DIMS: usize, T1: AsRef<GTensor<DIMS>>, T2: AsRef<GTensor<DIMS>>>(
    tensor1: T1,
    tensor2: T2,
) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor1
        .as_ref()
        .map_binary(tensor2, map_binop!(|el1, el2| el1.max(el2)))
}

pub fn sub_exp<const DIMS: usize, T1: AsRef<GTensor<DIMS>>, T2: AsRef<GTensor<DIMS>>>(
    tensor1: T1,
    tensor2: T2,
) -> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    tensor1
        .as_ref()
        .map_binary(tensor2, map_binop!(|el1, el2| (el1 - el2).exp()))
}
