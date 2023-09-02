extern crate alloc;
use crate::{
    shapes::{Shape, Unit},
    tensor::{
        cpu::{Cpu, CpuError},
        storage_traits::*,
    },
};
use alloc::sync::Arc;
use core::{any::TypeId, fmt};
use futures::executor::block_on;
use std::sync::RwLock;
use wgpu::{self, Buffer, BufferDescriptor, BufferUsages, Device, Instance, Queue};

use super::{resources::ResourceManager, LayoutType, WgpuVec};
// use crate::tensor::;

/// A GPU-accelerated device powered by [wgpu](https://docs.rs/wgpu/0.16.1/wgpu/index.html).
/// Depending on compilation targets, this device uses WebGPU (for targeting
/// WASM), Accelerate for MacOS, or Vulkan.
///
/// Data for tensors created by this device is not stored on the heap unless it
/// is mapped; instead, data only ever lives on VRAM. This differs from the Cuda
/// device, and has implications for read/write safety, `Cow` usage, and buffer caching.
#[derive(Debug, Clone)]
pub struct Wgpu {
    pub(crate) cpu: Cpu,
    instance: Arc<Instance>, // needed for poll_all
    pub(crate) dev: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) resources: Arc<RwLock<ResourceManager>>,
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
    /// Generic exception, contains a reason
    InvalidState(&'static str),
}

impl fmt::Display for WgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "device is out of memory"),
            Self::WrongNumElements => write!(f, "wrong number of elements"),
            Self::CannotCreate(why) => write!(f, "Cannot create Wgpu device: {}", why),
            Self::InvalidBufferUsage(why) => write!(f, "Invalid buffer usage: {}", why),
            Self::InvalidState(why) => write!(f, "Invalid state: {}", why),
        }
    }
}
#[cfg(not(feature = "no-std"))]
impl std::error::Error for WgpuError {}

impl From<CpuError> for WgpuError {
    fn from(value: CpuError) -> Self {
        match value {
            CpuError::OutOfMemory => Self::OutOfMemory,
            CpuError::WrongNumElements => Self::WrongNumElements,
        }
    }
}

impl HasErr for Wgpu {
    type Err = WgpuError;
}

impl Wgpu {
    /// Submit a set of commands to the GPU for execution.
    ///
    /// ## Example
    /// ```
    /// let dev: Wgpu = Default::default();
    /// let buf: WgpuVec<f32> = dev.try_alloc_len(64).unwrap();
    /// let data: Vec<f32> = vec![5.; 64];
    /// dev.submit(|encoder| {
    ///     // todo
    /// });
    /// ```
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

    pub(crate) fn get_or_register_pipeline<E: Unit>(
        &self,
        shader_name: &str,
        entrypoint_name: &str,
        shader_source: &str,
        layout_type: LayoutType,
    ) -> Result<Arc<wgpu::ComputePipeline>, WgpuError> {
        let resources = self.resources.read().map_err(|_| {
            WgpuError::InvalidState("could not lock device because lock was poisoned")
        })?;
        match resources.get_pipeline::<E>(shader_name, entrypoint_name) {
            Some(pipeline) => Ok(pipeline),
            None => {
                drop(resources);
                let mut resources = self.resources.write().map_err(|_| {
                    WgpuError::InvalidState("could not lock device because lock was poisoned")
                })?;
                Ok(resources.register_pipeline::<E>(
                    self.dev.as_ref(),
                    shader_name,
                    entrypoint_name,
                    shader_source,
                    layout_type,
                ))
            }
        }
    }
}

impl Default for Wgpu {
    fn default() -> Self {
        #[cfg(not(feature = "f16"))]
        let features: wgpu::Features = Default::default();
        #[cfg(feature = "f16")]
        let features: wgpu::Features = wgpu::Features::default() | wgpu::Features::SHADER_F16;

        let limits: wgpu::Limits = Default::default();
        let device_desc = wgpu::DeviceDescriptor {
            label: Some("dfdx"),
            features,
            limits,
        };
        let instance = Arc::new(Instance::new(Default::default()));
        let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
        let (dev, queue) = block_on(adapter.request_device(&device_desc, None)).unwrap();
        let resources = ResourceManager::new(&dev);

        Self {
            instance,
            dev: Arc::new(dev),
            queue: Arc::new(queue),
            cpu: Default::default(),
            resources: Arc::new(RwLock::new(resources)),
        }
    }
}

impl Synchronize for Wgpu {
    fn try_synchronize(&self) -> Result<(), WgpuError> {
        self.instance.poll_all(true);
        Ok(())
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

        // This buffer is mappable and doesn't need to be copied into a
        // mappable buffer. Usually only happens when
        // MAPPABLE_PRIMARY_BUFFERS is enabled.
        if usages.contains(BufferUsages::MAP_READ) {
            // Panics if buffer isn't immediately mappable
            let view = buffer.slice(..).get_mapped_range();
            let slice: &[E] = unsafe {
                let (prefix, slice, suffix) = view.align_to::<E>();
                assert_eq!(prefix.len(), 0);
                assert_eq!(suffix.len(), 0);
                slice
            };
            return slice.to_vec();

        // buffer is copyable on the gpu, so we copy it into a mappable
        // buffer and move it over
        } else if usages.contains(BufferUsages::COPY_SRC) {
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
            let slice: &[E] = unsafe {
                let (prefix, slice, suffix) = view.align_to::<E>();
                assert_eq!(prefix.len(), 0);
                assert_eq!(suffix.len(), 0);
                slice
            };
            return slice.to_vec();
        } else {
            panic!("cannot map wgpu buffer with usages: {:?}", usages);
        }
    }

    fn len(&self, v: &Self::Vec) -> usize {
        v.len()
    }
}
