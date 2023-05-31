use crate::shapes::{Shape, Unit};
use crate::tensor::storage_traits::*;
use core::any::TypeId;
use std::{
    sync::{Arc, Mutex, MutexGuard},
    vec::Vec,
};
use wgpu;
// use std::future::Future;
use futures::executor;

use super::vec::*;

/// A GPU-backed device leveraging [wgpu](https://wgpu.rs/), providing support
/// for the Accelerate framework on macOS and
/// [WebGPU](https://gpuweb.github.io/gpuweb/) in the browser.
#[derive(Debug, Clone)]
pub struct Wgpu {
    instance: Arc<wgpu::Instance>,
    pub(crate) dev: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
}

impl Default for Wgpu {
    fn default() -> Self {
        let instance: wgpu::Instance = Default::default();
        let device_future = async {
            // todo: request high-powered device
            let adapter = instance.request_adapter(&Default::default()).await.unwrap();
            adapter
                .request_device(&Default::default(), None)
                .await
                .unwrap()
        };
        let (dev, queue): (wgpu::Device, wgpu::Queue) = executor::block_on(device_future);
        return Wgpu::new(instance, dev, queue);
    }
}

impl Wgpu {
    pub fn new(instance: wgpu::Instance, dev: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            instance: Arc::new(instance),
            dev: Arc::new(dev),
            queue: Arc::new(queue),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum WgpuError {
    /// Device is out of memory
    OutOfMemory,
    /// Not enough elements were provided when creating a tensor
    WrongNumElements,
}
impl std::fmt::Display for WgpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfMemory => f.write_str("WgpuError::OutOfMemory"),
            Self::WrongNumElements => f.write_str("WgpuError::WrongNumElements"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WgpuError {}

impl HasErr for Wgpu {
    type Err = WgpuError;
}

impl<E: Unit> Storage<E> for Wgpu {
    type Vec = WgpuVec<E>;

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Self::Err> {
        let v = Self::Vec::new(self.dev.clone(), len, false);
        Ok(v)
        // let buf = self.dev.create_buffer(&wgpu::BufferDescriptor {
        //     label: None,
        //     // size: (len * E::NUM_BYTES) as u64,

        //     usage: wgpu::BufferUsages::STORAGE,
        //     mapped_at_creation: false,
        // });
        // Ok(buf)
    }

    fn len(&self, v: &Self::Vec) -> usize {
        v.len()
    }

    fn tensor_to_vec<S: Shape, T>(&self, tensor: &crate::prelude::Tensor<S, E, Self, T>) -> Vec<E> {
        // let data = *tensor.data;
        debug_assert_eq!(TypeId::of::<E>(), tensor.data.ty, "WgpuBuffer::as_slice(): type mismatch between buffer's contents and expected type");
        let view = tensor.data.view();
        let slice = unsafe {
            let (prefix, slice, suffix) = view.align_to::<E>();
            assert_eq!(prefix.len(), 0);
            assert_eq!(suffix.len(), 0);
            slice
        };
        slice.to_vec()
    }
}

impl Synchronize for Wgpu {
    fn try_synchronize(&self) -> Result<(), Self::Err> {
        self.instance.poll_all(true);
        Ok(())
    }
    fn synchronize(&self) {
        self.instance.poll_all(true);
    }
}
