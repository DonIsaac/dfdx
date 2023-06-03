use crate::shapes::{Shape, Unit};
use crate::tensor::storage_traits::*;
use core::any::TypeId;
use core::array::IntoIter;
use core::pin::Pin;
use core::slice;
use std::{
    // todo: what should we do when nostd feature is enabled?
    collections::BTreeMap,
    sync::{Arc, MutexGuard, RwLock},
    vec::Vec,
};
use wgpu;
// use std::future::Future;
use futures::executor;

use super::{
    resources::*,
    vec::*
};

type SafeBTreeMap<K, V> = Arc<RwLock<BTreeMap<K, V>>>;
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineKey {
    shader_name: &'static str,
    entrypoint: &'static str,
}
impl Into<PipelineKey> for (&'static str, &'static str) {
    fn into(self) -> PipelineKey {
        PipelineKey {
            shader_name: self.0,
            entrypoint: self.1,
        }
    }
}

#[derive(Debug)]
pub enum OpLayoutType {
    Unary,
    Binary,
    // todo: support more types of ops
    // Custom(
    //     &'static str,
    //     &'static wgpu::BindGroupLayoutDescriptor<'static>,
    // ),
}

/// A GPU-backed device leveraging [wgpu](https://wgpu.rs/), providing support
/// for the Accelerate framework on macOS and
/// [WebGPU](https://gpuweb.github.io/gpuweb/) in the browser.
#[derive(Debug, Clone)]
pub struct Wgpu {
    instance: Arc<wgpu::Instance>,
    pub(crate) dev: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) shaders: SafeBTreeMap<&'static str, wgpu::ShaderModule>,
    pub(crate) layouts: SafeBTreeMap<&'static str, Arc<wgpu::BindGroupLayout>>,
    pub(crate) pipelines: SafeBTreeMap<PipelineKey, Arc<wgpu::ComputePipeline>>, // pipeline_cache: Arc<RwLock<
    empty_uniform: Arc<wgpu::Buffer>,
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
        // let preexisting_layouts = &[
        //     (UNARY_OP_LAYOUT_NAME, dev.create_bind_group_layout(&unary_op_layout())),
        //     (BINARY_OP_LAYOUT_NAME, dev.create_bind_group_layout(&binary_op_layout())),
        // ];
        let empty_uniform = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("empty_uniform"),
            size: 0,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance = Self {
            instance: Arc::new(instance),
            dev: Arc::new(dev),
            queue: Arc::new(queue),
            shaders: Default::default(),
            layouts: Default::default(),
            pipelines: Default::default(),
            empty_uniform: Arc::new(empty_uniform)
        };

        {
            let mut layouts = instance.layouts.write().unwrap();
            layouts.insert(
                UNARY_OP_LAYOUT_NAME,
                Arc::new(instance.dev.create_bind_group_layout(&unary_op_layout())),
            );
            layouts.insert(
                BINARY_OP_LAYOUT_NAME,
                Arc::new(instance.dev.create_bind_group_layout(&binary_op_layout())),
            );
        }

        return instance
    }

    /// Checks if a shader module as already been loaded.
    pub(crate) fn has_module(&self, shader_name: &'static str) -> bool {
        self.shaders.read().unwrap().contains_key(shader_name)
    }

    /// Loads a shader module from source code. This is a blocking operation.
    pub(crate) fn load_module(
        &self,
        shader_name: &'static str,
        source_code: &'static str,
    ) {
        let source = wgpu::ShaderSource::Wgsl(source_code.into());
        let shader = self.dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_name),
            source,
        });
        let existing = self.shaders.write().unwrap().insert(shader_name, shader);
        debug_assert!(existing.is_none(), "Tried to load existing shader module");
    }

    pub(crate) fn get_layout(
        &self,
        layout_type: OpLayoutType
    ) -> Arc<wgpu::BindGroupLayout> {
        let layouts = self.layouts.read().unwrap();
        match layout_type {
            OpLayoutType::Unary => layouts.get(UNARY_OP_LAYOUT_NAME).unwrap().clone(),
            OpLayoutType::Binary => layouts.get(BINARY_OP_LAYOUT_NAME).unwrap().clone(),
        }
    }


    /// Checks if a compute pipeline has already been created.
    pub(crate) fn has_pipeline(
        &self,
        shader_name: &'static str,
        function_name: &'static str,
    ) -> bool {
        self.pipelines
            .read()
            .unwrap()
            .contains_key(&(shader_name, function_name).into())
    }

    pub(crate) fn get_pipeline(
        &self,
        shader_name: &'static str,
        function_name: &'static str,
    ) -> Option<Arc<wgpu::ComputePipeline>> {
        self.pipelines
            .read()
            .unwrap()
            .get(&(shader_name, function_name).into())
            .map(|p| p.clone())
    }

    pub(crate) fn load_pipeline(
        &self,
        shader_name: &'static str,
        function_name: &'static str,
        layout: &wgpu::BindGroupLayout,
    ) {
        let shaders = self.shaders.read().unwrap();
        let module = shaders.get(shader_name).unwrap();
        let pipeline_layout = self
            .dev
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(function_name),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .dev
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(function_name),
                layout: Some(&pipeline_layout),
                module,
                entry_point: function_name,
            });
        self.pipelines
            .write()
            .unwrap()
            .insert((shader_name, function_name).into(), Arc::new(pipeline));
    }

    pub(crate) fn execute_op(
        &self,
        pipeline: &wgpu::ComputePipeline,
        params: &wgpu::BindGroup,
        label: Option<&'static str>,
        work_groups: &WorkGroupSize,
    ) -> wgpu::SubmissionIndex {
        let mut encoder = self
            .dev
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label });
        {
            let (x, y, z) = *work_groups;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, params, &[]);
            pass.dispatch_workgroups(x, y, z);
        }
        let cmd = [encoder.finish()];
        let submission_index = self.queue.submit(cmd);
        return submission_index;
    }

    pub(crate) fn empty_metadata(&self) -> Pin<&wgpu::Buffer> {
        Pin::new(&self.empty_uniform)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum WgpuError {
    /// Device is out of memory
    OutOfMemory,
    /// Not enough elements were provided when creating a tensor
    WrongNumElements,
    /// Attempted to create a new GPU resource (pipeline, shader module, etc)
    /// when one already exists
    ResourceAlreadyExists(&'static str),
}

impl std::fmt::Display for WgpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfMemory => f.write_str("WgpuError::OutOfMemory"),
            Self::WrongNumElements => f.write_str("WgpuError::WrongNumElements"),
            Self::ResourceAlreadyExists(resource_name) => {
                write!(f, "WgpuError::ResourceAlreadyExists ({})", resource_name)
            }
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
        debug_assert_eq!(
            TypeId::of::<E>(),
            tensor.data.ty,
            "WgpuBuffer::as_slice(): type mismatch between buffer's contents and expected type"
        );
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
