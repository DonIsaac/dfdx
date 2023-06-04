pub(crate) const UNARY_OP_LAYOUT_NAME: &'static str = "unary";
pub(crate) const BINARY_OP_LAYOUT_NAME: &'static str = "binary";

pub(crate) type WorkGroupSize = (u32, u32, u32);
// #[derive(Debug, Clone, Copy)]
// pub(crate) struct WorkGroupSize(u32, u32, u32);
// impl Into<(u32, u32, u32)> for WorkGroupSize {
//     fn into(self) -> (u32, u32, u32) {
//         (self.0, self.1, self.2)
//     }
// }
// impl Into<&(u32, u32, u32)> for &WorkGroupSize {
//     fn into(self) -> &'static (u32, u32, u32) {
//         &(self.0, self.1, self.2)
//     }
// }

pub(crate) const fn unary_op_layout() -> wgpu::BindGroupLayoutDescriptor<'static> {
    const entries: [wgpu::BindGroupLayoutEntry; 3] = [
        // input tensor buffer
        storage_entry(0, true),
        // metadata buffer
        // uniform_entry(1),
        storage_entry(1, true),
        // output tensor buffer
        storage_entry(2, false)
    ];
    wgpu::BindGroupLayoutDescriptor {
        label: Some(UNARY_OP_LAYOUT_NAME),
        entries: &entries
    }
}

pub(crate) const fn binary_op_layout() -> wgpu::BindGroupLayoutDescriptor<'static> {
    const entries: [wgpu::BindGroupLayoutEntry; 4] = [
        // lhs tensor buffer
        storage_entry(0, true),
        // rhs tensor buffer
        storage_entry(1, true),
        // metadata buffer
        // uniform_entry(2),
        storage_entry(2, true),
        // output tensor buffer
        storage_entry(3, false)
    ];
    wgpu::BindGroupLayoutDescriptor {
        label: Some(BINARY_OP_LAYOUT_NAME),
        entries: &entries
    }
}

/// Creates a [`wgpu::BindGroupLayoutEntry`] for a storage buffer. Useful for
/// composing a [`wgpu::BindGroupLayout`].
const fn storage_entry(index: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: index,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None
        },
        count: None
    }
}

/// Creates a [`wgpu::BindGroupLayoutEntry`] for a Uniform buffer.
/// 
/// Uniforms are always read-only
const fn uniform_entry(index: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: index,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None
        },
        count: None
    }
}
