extern crate alloc;
use crate::shapes::{Shape, Unit};
use crate::tensor::storage_traits::*;
use alloc::sync::Arc;
use core::{
    any::TypeId,
    fmt
};
use wgpu::{self, Buffer, BufferDescriptor, BufferUsages, Device, Instance, Queue};
use futures::executor::block_on;

use crate::tensor::cpu::{Cpu, CpuError};
use super::WgpuVec;

#[derive(Debug, Clone)]
pub struct Wgpu {
    pub(crate) cpu: Cpu,
    instance: Arc<Instance>, // needed for poll_all
    pub(crate) dev: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
}
static_assertions::assert_impl_all!(Wgpu: Send, Sync);

#[derive(Debug, Clone)]
pub enum WgpuError {
    /// Device is out of memory
    OutOfMemory,
    WrongNumElements,
    /// Failed to create the device, either because WebGPU isn't available or a
    /// device with the desired features/limits isn't available
    CannotCreate(String),
    /// Tried to use a buffer for something that its usages won't support
    InvalidBufferUsage(String),
}

impl fmt::Display for WgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "device is out of memory"),
            Self::WrongNumElements => write!(f, "wrong number of elements"),
            Self::CannotCreate(why) => write!(f, "Cannot create Wgpu device: {}", why),
            Self::InvalidBufferUsage(why) => write!(f, "Invalid buffer usage: {}", why)
        }
    }
}
#[cfg(not(feature = "no-std"))]
impl std::error::Error for WgpuError {}

impl From<CpuError> for WgpuError {
    fn from(value: CpuError) -> Self {
        match value {
            CpuError::OutOfMemory => Self::OutOfMemory,
            CpuError::WrongNumElements => Self::WrongNumElements
        }
    }
}

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
            cpu: Default::default()
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
        Ok(WgpuVec::storage(Arc::new(self.clone()), len))
    }

    fn tensor_to_vec<S: Shape, T>(&self, tensor: &crate::prelude::Tensor<S, E, Self, T>) -> Vec<E> {
        let v: &WgpuVec<E> = tensor.data.as_ref();
        debug_assert_eq!(TypeId::of::<E>(), v.ty);
        let buffer = &v.buffer;
        let usages = buffer.usage();

        match usages {
            // This buffer is mappable and doesn't need to be copied into a
            // mappable buffer. Usually only happens when
            // MAPPABLE_PRIMARY_BUFFERS is enabled.
            BufferUsages::MAP_READ => {
                // Panics if buffer isn't immediately mappable
                let view = buffer.slice(..).get_mapped_range();
                let mut v: Vec<E> = Vec::with_capacity(buffer.size() as usize);
                unsafe {
                    let (prefix, slice, suffix) = view.align_to::<E>();
                    assert_eq!(prefix.len(), 0);
                    assert_eq!(suffix.len(), 0);
                    v.copy_from_slice(slice);
                }
                return v;
            }

            // buffer is copyable on the gpu, so we copy it into a mappable
            // buffer and move it over
            BufferUsages::COPY_SRC => {
                let new_buffer = self.dev.create_buffer(&BufferDescriptor {
                    label: None,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    size: buffer.size(),
                    mapped_at_creation: false,
                });
                let copy_id = self.submit(|encoder| {
                    encoder.copy_buffer_to_buffer(buffer, 0, &new_buffer, 0, buffer.size())
                });
                new_buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |res| res.unwrap());
                self.dev
                    .poll(wgpu::Maintain::WaitForSubmissionIndex(copy_id));

                let view = new_buffer.slice(..).get_mapped_range();
                let mut v: Vec<E> = Vec::with_capacity(new_buffer.size() as usize);
                unsafe {
                    let (prefix, slice, suffix) = view.align_to::<E>();
                    assert_eq!(prefix.len(), 0);
                    assert_eq!(suffix.len(), 0);
                    v.copy_from_slice(slice);
                }
                return v;
            }
            _ => {
                panic!("cannot map wgpu buffer with usages: {:?}", usages);
            }
        }
    }

    fn len(&self, v: &Self::Vec) -> usize {
        v.len()
    }
}
