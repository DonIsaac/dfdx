use crate::shapes::Unit;
use core::hash::{Hasher, BuildHasher};
use std::{collections::{BTreeMap, hash_map::DefaultHasher}, sync::Arc};
use wgpu::{
    self,
    // layouts
    BindGroup,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    ComputePipeline,
    PipelineLayout,
    PipelineLayoutDescriptor,
    ShaderModule,
    // kernels
    ShaderSource,
};

// pub(crate) type PipelineKey = &'static str;
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PipelineKey(u64);

#[derive(Debug)]
pub(crate) struct ResourceManager {
    binary_layout: BindGroupLayout,
    unary_layout: BindGroupLayout,
    pipelines: BTreeMap<PipelineKey, Arc<ComputePipeline>>,
}

impl ResourceManager {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let binary_layout = device.create_bind_group_layout(&create_binary_layout_descriptor());
        let unary_layout = device.create_bind_group_layout(&create_unary_layout_descriptor());
        Self {
            binary_layout,
            unary_layout,
            pipelines: Default::default()
        }
    }
    /// Get a compute pipeline for an operation.
    /// 
    /// Returns None if the pipeline hasn't been registered yet.
    pub(crate) fn get_pipeline<E: Unit>(
        &self,
        shader_name: &'static str,
        entrypoint_name: &'static str,
    ) -> Option<Arc<ComputePipeline>> {
        let key = self.to_key::<E, DefaultHasher>(shader_name, entrypoint_name);
        self.pipelines.get(&key).map(|o| o.clone())
    }

    fn to_key<E: Unit, H: Hasher + Default>(&self, shader_name: &'static str, entrypoint_name: &'static str) -> PipelineKey {
        let mut hasher: H = Default::default();
        hasher.write(shader_name.as_bytes());
        hasher.write(entrypoint_name.as_bytes());
        hasher.write(E::NAME.as_bytes());
        let key = hasher.finish();
        return PipelineKey(key);
    }

}

const fn create_binary_layout_descriptor() -> BindGroupLayoutDescriptor<'static> {
    const entries: [BindGroupLayoutEntry; 4] = [
        /// lhs (tensor)
        storage_entry(0, true),
        /// rhs (tensor)
        storage_entry(1, true),
        /// metadata/operation details
        storage_entry(2, true),
        /// output (tensor)
        storage_entry(3, false)
    ];
    BindGroupLayoutDescriptor {
        label: Some("binary layout"),
        entries: &entries
    }
}
const fn create_unary_layout_descriptor() -> BindGroupLayoutDescriptor<'static> {
    const entries: [BindGroupLayoutEntry; 3] = [
        /// input (tensor)
        storage_entry(0, true),
        /// metadata/operation details
        storage_entry(1, true),
        /// output (tensor)
        storage_entry(2, false)
    ];
    BindGroupLayoutDescriptor {
        label: Some("unary layout"),
        entries: &entries
    }
}
const fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
