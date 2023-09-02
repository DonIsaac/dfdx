extern crate alloc;
use alloc::borrow::Cow;
use std::sync::RwLock;
use num_traits::AsPrimitive;
use ::wgpu;
use wgpu::{
    BindGroup,
    BindGroupDescriptor,
    BindGroupLayout
};
use crate::{
    shapes::{Shape, Unit},
    tensor_ops::utilities::wgpu_kernels::params,
    tensor::{
        Storage,
        Tensor,
        wgpu::{Wgpu, WgpuError, WgpuVec, LayoutType}
    }, prelude::wgpu::WgpuNativeType
};

const ENTRYPOINT_NAME: &str = "main";
const KERNEL: &str = "
alias T = __SRC__;
alias U = __DST__;

@group(0) @binding(0)
var<storage, read> in: array<T>;
@group(0) @binding(1)
var<storage, read_write> out: array<U>;
@compute @workgroup_size(1, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;
    out[i] = U(in[i]);
}
";
const LAYOUT_DESC: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
    label: Some("to-dtype"),
    entries: &[
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None,
        },
    ],
};

impl<E1: WgpuNativeType + AsPrimitive<E2>, E2: WgpuNativeType> super::ToDtypeKernel<E1, E2> for Wgpu {
    fn forward<S: Shape>(inp: Tensor<S, E1, Self>) -> Result<Tensor<S, E2, Self>, Self::Err> {
        // static mut LAYOUT: RwLock<Option<wgpu::BindGroupLayout>> = RwLock::new(None);
        // todo: this doesn't fit in with the type template used by
        // ResourceManager. Should probably re-think that. For now, we'll just
        // use f32
        let module_name = std::format!("convert_{}_to_{}", E1::NAME, E2::NAME);
        let dev = inp.device;

        let pipeline = {
            let resources = dev.resources.read().map_err(|_| {
                WgpuError::InvalidState("could not lock device because lock was poisoned")
            })?;
            match resources.get_pipeline::<f32>(module_name.as_ref(), ENTRYPOINT_NAME) {
                Some(pipeline) => pipeline,
                None => {
                    drop(resources);
                    let mut resources = dev.resources.write().map_err(|_| {
                        WgpuError::InvalidState("could not lock device because lock was poisoned")
                    })?;
                    let shader_source: String = 
                        KERNEL
                            .replace("__SRC__", E1::NAME)
                            .replace("__DST__", E2::NAME);
                    resources.register_pipeline::<f32>(
                        dev.dev.as_ref(),
                        module_name.as_ref(),
                        ENTRYPOINT_NAME,
                        shader_source.as_str(),
                        LayoutType::Dynamic,
                    )
                }
            }
        };
        let numel = inp.shape.num_elements();
        let shape = inp.shape;
        let strides = shape.strides();
        let output: WgpuVec<E2> = dev.try_alloc_len(numel)?;

        let params: BindGroup = {
            // let layout = unsafe {
            //     let lock = LAYOUT.read().unwrap();
            //     if lock.is_none() {
            //         drop(lock);
            //         let mut lock = LAYOUT.write().unwrap();
            //         let _ = lock.insert(dev.dev.create_bind_group_layout(&LAYOUT_DESC));
            //         drop(lock);
            //         LAYOUT.read().unwrap().unwrap()
            //         // lock.replace(value)
            //         // LAYOUT = Some(dev.dev.create_bind_group_layout(&LAYOUT_DESC));
            //     } else {
            //         lock.unwrap()
            //     }
            // };
            params!(dev, pipeline; inp.data, output)
        };
        let _idx = dev.submit(|encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("to-dtype") });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &params, &[]);
            pass.dispatch_workgroups(numel as u32, 1, 1);
        });
        Ok(dev.build_tensor(shape, strides, output))
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn test_i32_to_u32() {
        let dev: Wgpu = Default::default();
        let a = dev.tensor([[1, 2, 3], [4, 5, 6]]);
        let b = a.to_dtype::<u32>();
        let b = b.as_vec();
        assert_eq!(b, vec![1, 2, 3, 4, 5, 6])
    }
}
