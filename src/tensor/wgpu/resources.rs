use crate::shapes::Unit;
use core::hash::{BuildHasher, Hasher};
use std::{
    collections::{hash_map::DefaultHasher, BTreeMap},
    sync::Arc,
};
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

#[derive(Debug, Copy, Clone)]
pub(crate) enum LayoutType {
    Binary,
    Unary,
    Dynamic,
}

#[cfg(all(feature = "f16", nightly))]
const SHADER_F16_PREFIX: &str = "enable f16;\n";

#[derive(Debug)]
pub(crate) struct ResourceManager {
    binary_layout: BindGroupLayout,
    unary_layout: BindGroupLayout,
    pipelines: BTreeMap<PipelineKey, Arc<ComputePipeline>>,
}

impl ResourceManager {
    /// This string will be found/replaced with an actual data type name inside
    /// of shader source code. Used for a poor man's generic type system.
    pub(crate) const TYPENAME_TEMPLATE: &'static str = "__TYPENAME__";

    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let binary_layout = device.create_bind_group_layout(&create_binary_layout_descriptor());
        let unary_layout = device.create_bind_group_layout(&create_unary_layout_descriptor());
        Self {
            binary_layout,
            unary_layout,
            pipelines: Default::default(),
        }
    }
    /// Get a compute pipeline for an operation.
    ///
    /// Returns None if the pipeline hasn't been registered yet.
    pub(crate) fn get_pipeline<E: Unit>(
        &self,
        shader_name: &str,
        entrypoint_name: &str,
    ) -> Option<Arc<ComputePipeline>> {
        let key = self.to_key::<E, DefaultHasher>(shader_name, entrypoint_name);
        self.pipelines.get(&key).map(|o| o.clone())
    }
    pub(crate) fn has_pipeline<E: Unit>(
        &self,
        shader_name: &str,
        entrypoint_name: &str,
    ) -> bool {
        let key = self.to_key::<E, DefaultHasher>(shader_name, entrypoint_name);
        self.pipelines.contains_key(&key)
    }

    pub(crate) fn register_pipeline<E: Unit>(
        &mut self,
        device: &wgpu::Device,
        shader_name: &str,
        entrypoint_name: &str,
        shader_source: &str,
        layout_type: LayoutType,
    ) -> Arc<ComputePipeline> {
        // compose pipeline layout
        let layout: Option<PipelineLayout> = {
            let bind_group_layout = match layout_type {
                LayoutType::Unary => Some(&self.unary_layout),
                LayoutType::Binary => Some(&self.binary_layout),
                LayoutType::Dynamic => None,
            };
            bind_group_layout.map(|l| {
                device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[l],
                    push_constant_ranges: &[], // todo
                })
            })
        };
        // compile shader source and replace template types with E
        let module: ShaderModule = {
            #[cfg(nightly)] {
                #[cfg(not(feature = "f16"))]
                let templated_source = shader_source.replace(Self::TYPENAME_TEMPLATE, E::NAME);
                #[cfg(feature = "f16")]
                let templated_source = SHADER_F16_PREFIX.to_owned() + shader_source.replace(Self::TYPENAME_TEMPLATE, E::NAME).as_str();
            }
            #[cfg(not(nightly))]
            let templated_source = shader_source.replace(Self::TYPENAME_TEMPLATE, E::NAME);
            let templated_source_str = templated_source.as_str();
            let source = ShaderSource::Wgsl(templated_source_str.into());
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(shader_name),
                source,
            })
        };

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entrypoint_name),
            layout: layout.as_ref(),
            module: &module,
            entry_point: entrypoint_name,
        });
        let pipeline = Arc::new(pipeline);

        let key = self.to_key::<E, DefaultHasher>(shader_name, entrypoint_name);
        self.pipelines.insert(key, pipeline.clone());

        return pipeline;
    }

    fn to_key<E: Unit, H: Hasher + Default>(
        &self,
        shader_name: &str,
        entrypoint_name: &str,
    ) -> PipelineKey {
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
        // lhs (tensor)
        storage_entry(0, true),
        // rhs (tensor)
        storage_entry(1, true),
        // metadata/operation details
        storage_entry(2, true),
        // output (tensor)
        storage_entry(3, false),
    ];
    BindGroupLayoutDescriptor {
        label: Some("binary layout"),
        entries: &entries,
    }
}
const fn create_unary_layout_descriptor() -> BindGroupLayoutDescriptor<'static> {
    const entries: [BindGroupLayoutEntry; 3] = [
        // input (tensor)
        storage_entry(0, true),
        // metadata/operation details
        storage_entry(1, true),
        // output (tensor)
        storage_entry(2, false),
    ];
    BindGroupLayoutDescriptor {
        label: Some("unary layout"),
        entries: &entries,
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
