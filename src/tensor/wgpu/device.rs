extern crate alloc;
use crate::shapes::{Shape, Unit};
use crate::tensor::storage_traits::*;
use alloc::sync::Arc;
use core::fmt;
use wgpu::{self, Buffer, Device, Instance, Queue};

use super::WgpuVec;
use futures::executor::block_on;

#[derive(Debug, Clone)]
pub struct Wgpu {
    instance: Arc<Instance>, // needed for poll_all
    dev: Arc<Device>,
    queue: Arc<Queue>,
}
static_assertions::assert_impl_all!(Wgpu: Send, Sync);

#[derive(Debug, Clone)]
pub enum WgpuError {
    CannotCreate(String),
}
impl fmt::Display for WgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WgpuError::CannotCreate(s) => write!(f, "Cannot create Wgpu device: {}", s),
        }
    }
}
#[cfg(not(feature = "no-std"))]
impl std::error::Error for WgpuError {}

impl HasErr for Wgpu {
    type Err = WgpuError;
}

impl Wgpu {
    pub(crate) fn submit<F>(&self, command_builder: F) -> wgpu::SubmissionIndex
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        let mut encoder = self
            .dev
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        command_builder(&mut encoder);
        let cmd = [encoder.finish()];
        self.queue.submit(cmd)
    }
}
impl Default for Wgpu {
    fn default() -> Self {
        let instance = Arc::new(Instance::new(Default::default()));
        let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
        let (dev, queue) = block_on(adapter.request_device(&Default::default(), None)).unwrap();
        Self {
            instance,
            dev: Arc::new(dev),
            queue: Arc::new(queue),
        }
    }
}

impl Synchronize for Wgpu {
    fn try_synchronize(&self) -> Result<(), WgpuError> {
        self.instance.poll_all(true);
        Ok(())
        // self.dev.synchronize().map_err(CudaError::from)
    }
}

impl<E: Unit> Storage<E> for Wgpu {
    type Vec = WgpuVec<E>;

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Self::Err> {
        todo!()
    }

    fn tensor_to_vec<S: Shape, T>(&self, tensor: &crate::prelude::Tensor<S, E, Self, T>) -> Vec<E> {
        todo!()
    }

    fn len(&self, v: &Self::Vec) -> usize {
        v.len()
    }
}
