use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use anyhow::{bail, ensure, Error, Ok, Result};
use memmap2::Mmap;
use safetensors::{tensor as st, SafeTensors};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorType {
    BFloat16,
}

pub struct MmChunkHolder<'a> {
    soffs: usize,
    eoffs: usize,
    mmap: &'a Mmap,
}

#[cfg(unix)]
impl<'a> Drop for MmChunkHolder<'a> {
    fn drop(&mut self) {
        let soffs = self.mmap.as_ptr() as usize + self.soffs;
        self.mmap
            .advise_range(memmap2::Advice::DontNeed, soffs, self.eoffs - self.soffs)
            .ok();
    }
}

#[derive(Clone)]
pub struct TensorData<'a> {
    pub name: String,
    pub dtype: TensorType,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
    // The TensorData could get cloned, so we want to
    // make sure that the Drop handler doesn't run until
    // all copies are gone.
    pub mmap: Option<Arc<MmChunkHolder<'a>>>,
}

#[derive(Clone)]
pub struct TensorDataMap<'a>(pub HashMap<String, TensorData<'a>>);

impl<'a> Deref for TensorDataMap<'a> {
    type Target = HashMap<String, TensorData<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for TensorDataMap<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> IntoIterator for TensorDataMap<'a> {
    type Item = (String, TensorData<'a>);

    type IntoIter = std::collections::hash_map::IntoIter<String, TensorData<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// This trait implementation converts a tuple of SafeTensors metadata
/// plus slice of bytes to a [TensorDataMap] which can be used to load the model.
// impl<'a> TryFrom<(st::Metadata, &'a [u8])> for TensorDataMap<'a> {
//     type Error = Error;

//     fn try_from((metadata, data): (st::Metadata, &'a [u8])) -> Result<Self, Self::Error> {
//         Ok(Self(
//             metadata
//                 .tensors()
//                 .into_iter()
//                 .map(|(name, ti)| {
//                     ensure!(
//                         ti.dtype == st::Dtype::BF16,
//                         "Only BFloat16 tensors supported currently"
//                     );
//                     let isize = 2; // 16bits
//                     let (soffs, eoffs) = ti.data_offsets;
//                     let bcount: usize = ti.shape.iter().product::<usize>() * isize;
//                     ensure!(bcount == (eoffs - soffs), "Unexpected tensor length.");

//                     Ok((
//                         name.clone(),
//                         TensorData {
//                             name,
//                             dtype: TensorType::BFloat16,
//                             shape: ti.shape.clone(),
//                             data: &data[soffs..eoffs],
//                             mmap: None,
//                         },
//                     ))
//                     //
//                 })
//                 .collect::<Result<HashMap<_, _>>>()?,
//         ))
//     }
//     //
// }

pub struct TensorDataFromSafeTensors<'a> {
    pub metadata: st::Metadata,
    pub mmap: &'a Mmap,
    pub offset: usize,
}

impl<'a> TryFrom<TensorDataFromSafeTensors<'a>> for TensorDataMap<'a> {
    type Error = Error;

    fn try_from(
        TensorDataFromSafeTensors {
            metadata,
            mmap,
            offset,
        }: TensorDataFromSafeTensors<'a>,
    ) -> Result<Self, Self::Error> {
        Ok(Self(
            metadata
                .tensors()
                .into_iter()
                .map(|(name, ti)| {
                    ensure!(
                        ti.dtype == st::Dtype::BF16,
                        "Only BFloat16 tensors supported currently"
                    );

                    let isize = 2; // 16bits
                    let (soffs, eoffs) = (ti.data_offsets.0 + offset, ti.data_offsets.1 + offset);
                    let bcount: usize = ti.shape.iter().product::<usize>() * isize;
                    ensure!(bcount == (eoffs - soffs), "Unexpected tensor length.");

                    Ok((
                        name.clone(),
                        TensorData {
                            name,
                            dtype: TensorType::BFloat16,
                            shape: ti.shape.clone(),
                            data: &mmap[soffs..eoffs],
                            mmap: Some(Arc::new(MmChunkHolder { soffs, eoffs, mmap })),
                        },
                    ))
                    //
                })
                .collect::<Result<HashMap<_, _>>>()?,
        ))
    }
    //
}

#[cfg(feature = "torch")]
fn repugnant_load(filename: String, mmap: &Mmap) -> Result<TensorDataMap<'_>> {
    use repugnant_pickle as rp;
    let tensors = rp::torch::RepugnantTorchTensors::new_from_file(filename)?;
    Ok(TensorDataMap(
        tensors
            .into_iter()
            .map(|rt| {
                ensure!(
                    rt.tensor_type == rp::TensorType::BFloat16,
                    "Only BFloat 16 tensors supported currently"
                );
                let bcount = rt.shape.iter().product::<usize>() * rt.tensor_type.size();
                let soffs = rt.absolute_offset as usize;
                let eoffs = soffs + bcount;

                Ok((
                    rt.name.clone(),
                    TensorData {
                        name: rt.name,
                        dtype: TensorType::BFloat16,
                        shape: rt.shape,
                        data: &mmap[soffs..eoffs],
                        mmap: Some(Arc::new(MmChunkHolder { soffs, eoffs, mmap })),
                    },
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?,
    ))
}

#[cfg(not(feature = "torch"))]
fn repugnant_load(_filename: String, _data: &[u8]) -> Result<TensorDataMap<'_>> {
    bail!("We're not compiled with PyTorch model file support. :(");
}

/// This trait implementation converts a tuple of filename + mmap to a
/// [TensorDataMap] which can be used to load the model.
impl<'a> TryFrom<(String, &'a Mmap)> for TensorDataMap<'a> {
    type Error = Error;

    fn try_from((filename, mmap): (String, &'a Mmap)) -> Result<Self, Self::Error> {
        if filename.ends_with(".st") || filename.ends_with(".safetensors") {
            let (headersize, md) = SafeTensors::read_metadata(&mmap[..])?;
            // Why is it necessary to randomly add 8 here? Because FU, that's why.
            // This is an undocumented requirement as far as I can tell.
            TensorDataFromSafeTensors {
                metadata: md,
                mmap,
                offset: headersize + 8,
            }
            .try_into()
        } else if filename.ends_with(".pth") || filename.ends_with(".pt") {
            repugnant_load(filename, mmap)
        } else {
            bail!("Unknown file type")
        }
    }
}
