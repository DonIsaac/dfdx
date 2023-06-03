// use wgpu::util::DeviceExt;
use crate::tensor::webgpu::{OpLayoutType, Wgpu};
use crate::{
    prelude::webgpu::resources::WorkGroupSize,
    shapes::{Dtype, Shape},
    tensor::*,
    // tensor::webgpu::WebGpu,
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};
use core::borrow::Borrow;
// use std::borrow::Cow;
extern crate alloc;
use alloc::{borrow::Cow, sync::Arc};
/// todo: make this a const fn, maybe using SmallVec
fn to_entries<'a>(buffers: &'a [&wgpu::Buffer]) -> Vec<wgpu::BindGroupEntry<'a>> {
    buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect()
}

pub trait WgpuKernel {
    fn layout(&self) -> wgpu::BindGroupLayoutDescriptor;
}

pub trait UnaryOpWgpuKernel<E> {
    const DF_USES_FX: bool;
    const HAS_CONST_DF: bool;

    /// WGSL source code for the kernel
    const WGSL_SRC: &'static str;

    /// Unique name for the kernel
    const MODULE_NAME: &'static str;

    /// Name of function in the .wgsl file (used as entrypoint)
    const FWD_FN_NAME: &'static str;

    /// Name of function in the .ggsl file (used as entrypoint)
    const BWD_FN_NAME: &'static str;

    const ALL_FN_NAMES: [&'static str; 2] = [Self::FWD_FN_NAME, Self::BWD_FN_NAME];
}

macro_rules! wgpu_unary {
    ($Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::wgpu_kernels::UnaryOpWgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = false;
            const HAS_CONST_DF: bool = false;
            const WGSL_SRC: &'static str = include_str!($Wgsl);
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
    (df(f(x)) $Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::wgpu_kernels::UnaryOpWgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = true;
            const HAS_CONST_DF: bool = false;
            const WGSL_SRC: &'static str = include_str!($Wgsl);
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
    (const_df() $Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd:tt) => {
        impl crate::tensor_ops::wgpu_kernels::UnaryOpWgpuKernel<$TypeName> for $Op {
            const DF_USES_FX: bool = false;
            const HAS_CONST_DF: bool = true;
            const WGSL_SRC: &'static str = include_str!($Wgsl);
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_FN_NAME: &'static str = $Bwd;
        }
    };
}

pub(crate) use wgpu_unary;

impl<E: Dtype, K: UnaryOpWgpuKernel<E>> UnaryKernel<K, E> for Wgpu {
    const BACKWARD_WITHOUT_INP: bool = K::DF_USES_FX;
    const BACKWARD_WITHOUT_DATA: bool = K::HAS_CONST_DF;

    fn forward<S: Shape>(
        &self,
        op: K,
        inp: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        // if !self.dev.has_func(K::MODULE_NAME, K::FWD_FN_NAME) {
        //     self.dev
        //         .load_ptx(K::PTX_SRC.into(), K::MODULE_NAME, &K::ALL_FN_NAMES)?;
        // }

        let layout = self.get_layout(OpLayoutType::Unary);
        // Try to just get the pipeline first. Should make the cached case as
        // fast as possible.
        let fwd_pipeline: Arc<wgpu::ComputePipeline> =
            match self.get_pipeline(K::MODULE_NAME, K::FWD_FN_NAME) {
                Some(pipeline) => pipeline,
                None => {
                    if !self.has_module(K::MODULE_NAME) {
                        self.load_module(K::WGSL_SRC, K::MODULE_NAME);
                    }
                    self.load_pipeline(K::MODULE_NAME, K::FWD_FN_NAME, layout.as_ref());
                    self.get_pipeline(K::MODULE_NAME, K::FWD_FN_NAME).unwrap()
                }
            };
        match inp {
            Cow::Borrowed(inp) => {
                let numel = inp.data.len();
                let storage = self.create_vec::<E>(numel);

                let cfg: WorkGroupSize = (numel as u32, 1, 1);
                let layout = self.get_layout(OpLayoutType::Unary);
                let params = self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: layout.as_ref(),
                    entries: to_entries(&[
                        inp.data.as_ref().into(),
                        self.empty_metadata().borrow(),
                        (&storage).into(),
                    ])
                    .as_slice(),
                });
                let _idx =
                    self.execute_op(fwd_pipeline.as_ref(), &params, Some(K::MODULE_NAME), &cfg);
                Ok(self.build_tensor(inp.shape, inp.strides, storage))
                // todo: record submission index for later synchronization

                // let cfg = launch_cfg::<128>(numel as u32);
                // let params = (op, numel, inp.data.as_ref(), &mut storage);
                // unsafe { fwd_fn.launch(cfg, params) }?;
                // Ok(self.build_tensor(inp.shape, inp.strides, storage))
            }
            Cow::Owned(mut inp) => {
                todo!()
                // inp.id = unique_id();
                // let numel = inp.data.len();
                // let cfg = launch_cfg::<128>(numel as u32);
                // let params = (op, numel, 0u64, Arc::make_mut(&mut inp.data));
                // unsafe { fwd_fn.launch(cfg, params) }?;
                // Ok(inp)
            }
        }
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        inp: &impl Tensorlike<S, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &impl Tensorlike<S, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
