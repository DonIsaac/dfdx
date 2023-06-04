// use wgpu::util::DeviceExt;
use crate::{
    shapes::{Dtype, Shape},
    tensor::webgpu::{OpLayoutType, Wgpu, WgpuVec, WorkGroupSize},
    tensor::*,
    // tensor::webgpu::WebGpu,
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};
use bytemuck::{Pod, Zeroable};
use core::{borrow::Borrow, mem};
use static_assertions;
use wgpu::util::DeviceExt;
// use std::borrow::Cow;
extern crate alloc;
use alloc::{borrow::Cow, sync::Arc};

fn to_entries<'a, I>(bindings: I) -> Vec<wgpu::BindGroupEntry<'a>>
where
    I: IntoIterator<Item = wgpu::BindingResource<'a>>,
{
    bindings
        .into_iter()
        .enumerate()
        .map(|(i, binding)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: binding,
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
                    entries: to_entries([
                        inp.data.as_entire_binding(),
                        self.empty_metadata().as_entire_binding(),
                        storage.as_entire_binding(),
                    ])
                    .as_slice(),
                });

                // todo: record submission index for later synchronization
                let _idx = self.submit_basic_op(
                    fwd_pipeline.as_ref(),
                    &params,
                    Some(K::MODULE_NAME),
                    &cfg,
                );
                Ok(self.build_tensor(inp.shape, inp.strides, storage))
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

pub trait BinaryOpCudaKernel<E> {
    const HAS_CONST_DF: bool;

    /// WGSL source code for the kernel
    const WGSL_SRC: &'static str;

    /// Unique name for the kernel
    const MODULE_NAME: &'static str;

    /// Name of function in the .cu file
    const FWD_FN_NAME: &'static str;

    /// Name of function in the .cu file
    const BWD_LHS_FN_NAME: &'static str;

    /// Name of function in the .cu file
    const BWD_RHS_FN_NAME: &'static str;

    const ALL_FN_NAMES: [&'static str; 3] = [
        Self::FWD_FN_NAME,
        Self::BWD_LHS_FN_NAME,
        Self::BWD_RHS_FN_NAME,
    ];
}

macro_rules! wgpu_binary {
    ($Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd_Lhs:tt, $Bwd_Rhs:tt) => {
        impl crate::tensor_ops::wgpu_kernels::BinaryOpCudaKernel<$TypeName> for $Op {
            const HAS_CONST_DF: bool = false;
            const WGSL_SRC: &'static str = $Wgsl;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_LHS_FN_NAME: &'static str = $Bwd_Lhs;
            const BWD_RHS_FN_NAME: &'static str = $Bwd_Rhs;
        }
    };
    (const_df() $Op:path, $TypeName:ty, $Wgsl:tt, $Fwd:tt, $Bwd_Lhs:tt, $Bwd_Rhs:tt) => {
        impl crate::tensor_ops::wgpu_kernels::BinaryOpCudaKernel<$TypeName> for $Op {
            const HAS_CONST_DF: bool = true;
            const WGSL_SRC: &'static str = $Wgsl;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_LHS_FN_NAME: &'static str = $Bwd_Lhs;
            const BWD_RHS_FN_NAME: &'static str = $Bwd_Rhs;
        }
    };
}

pub(crate) use wgpu_binary;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct BinaryKernelMeta<S: Shape> {
    numel: u32,
    num_dims: u32,
    /// contains lhs_shape, lhs_strides, rhs_strides in that order.
    /// This must be the last field in the struct
    info: [S::Concrete; 3],
}
// kernel info will be stored in a uniform buffer and so must be 16 byte aligned
static_assertions::const_assert!(core::mem::align_of::<BinaryKernelMeta<[usize; 2]>>() == 8);
impl<S: Shape> BinaryKernelMeta<S> {
    const CONCRETE_SIZE: usize = core::mem::size_of::<S::Concrete>();
    fn new(lhs: &Cow<Tensor<S, impl Dtype, Wgpu>>, rhs: &Cow<Tensor<S, impl Dtype, Wgpu>>) -> Self {
        let shape = lhs.shape;
        let numel = shape.num_elements();

        Self {
            numel: numel as u32,
            num_dims: S::NUM_DIMS as u32,
            info: [shape.concrete(), lhs.strides, rhs.strides],
        }
    }
    fn to_bytes(&self) -> Vec<u8> {
        let capacity =
            // numel, num_dims
            (mem::size_of::<u32>() * 2) +
            // info
            (Self::CONCRETE_SIZE * 3);
        debug_assert!(
            capacity % 8 == 0,
            "binary kernel meta is not 8 byte aligned"
        );
        let mut bytes = Vec::with_capacity(capacity);
        // Note: WebGPU uses little endian, but native devices may not
        // todo: check if running on webgpu, then use to_le_bytes
        // see: https://gpuweb.github.io/gpuweb/wgsl/#internal-value-layout
        bytes.extend_from_slice(&self.numel.to_ne_bytes());
        bytes.extend_from_slice(&self.num_dims.to_ne_bytes());
        bytes.extend_from_slice(bytemuck::bytes_of(&self.info));
        bytes
    }
}

impl<E: Dtype, K: BinaryOpCudaKernel<E> + Clone> BinaryKernel<K, E> for Wgpu {
    const BACKWARD_WITHOUT_DATA: bool = K::HAS_CONST_DF;
    fn forward<S: Shape>(
        &self,
        op: K,
        lhs: Cow<Tensor<S, E, Self>>,
        rhs: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let layout = self.get_layout(OpLayoutType::Binary);
        // Try to just get the pipeline first. Should make the cached case as
        // fast as possible.
        let fwd_pipeline: Arc<wgpu::ComputePipeline> =
            match self.get_pipeline(K::MODULE_NAME, K::FWD_FN_NAME) {
                Some(pipeline) => pipeline,
                None => {
                    if !self.has_module(K::MODULE_NAME) {
                        self.load_module(K::MODULE_NAME, K::WGSL_SRC);
                    }
                    self.load_pipeline(K::MODULE_NAME, K::FWD_FN_NAME, layout.as_ref());
                    self.get_pipeline(K::MODULE_NAME, K::FWD_FN_NAME).unwrap()
                }
            };

        // let shape = match &lhs {
        //     Cow::Borrowed(lhs) => lhs.shape,
        //     Cow::Owned(lhs) => lhs.shape,
        // };
        // let strides = shape.strides();
        // let numel = shape.num_elements();

        let shape = lhs.shape;
        let strides = shape.strides();
        let meta: BinaryKernelMeta<S> = BinaryKernelMeta::new(&lhs, &rhs);
        let cfg: WorkGroupSize = (meta.numel as u32, 1, 1);
        let meta_buffer = self
            .dev
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("binary kernel meta"),
                contents: &meta.to_bytes(),
                usage: wgpu::BufferUsages::STORAGE,
            });

        match (lhs, rhs) {
            (Cow::Borrowed(lhs), Cow::Borrowed(rhs)) => {
                let output: WgpuVec<E> = self.create_vec(meta.numel as usize);
                // let output = WgpuVec::new(self.dev.clone(), Some("output"), meta.numel as usize, wgpu::BufferUsages::STORAGE)
                let params: wgpu::BindGroup = {
                    let binding_entries = to_entries([
                        lhs.data.as_entire_binding(),
                        rhs.data.as_entire_binding(),
                        meta_buffer.as_entire_binding(),
                        output.as_entire_binding(),
                    ]);
                    self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("binary kernel params"),
                        layout: layout.as_ref(),
                        entries: binding_entries.as_slice(),
                    })
                };

                let _idx = self.submit_commands(|enc| {
                    // let pipeline = fwd_pipeline.clone();
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("binary kernel"),
                    });
                    pass.set_pipeline(fwd_pipeline.as_ref());
                    pass.set_bind_group(0, &params, &[]);
                    pass.dispatch_workgroups(cfg.0, cfg.1, cfg.2);
                });

                Ok(self.build_tensor(shape, strides, output))
            }
            (Cow::Owned(mut lhs), Cow::Owned(mut rhs)) => {
                let output = self.create_vec::<E>(meta.numel as usize);
                let lhs_valid = lhs.strides == lhs.shape.strides();
                let rhs_valid = rhs.strides == rhs.shape.strides();
                if lhs_valid || rhs_valid {
                    let lhs_count = std::sync::Arc::strong_count(&lhs.data);
                    let rhs_count = std::sync::Arc::strong_count(&rhs.data);
                    if rhs_valid && (rhs_count == 1 || !lhs_valid || lhs_count != 1) {
                        rhs.id = unique_id();
                        let params: wgpu::BindGroup = {
                            let binding_entries = to_entries([
                                lhs.data.as_entire_binding(),
                                rhs.data.as_entire_binding(),
                                meta_buffer.as_entire_binding(),
                                output.as_entire_binding(),
                            ]);
                            self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("binary kernel params"),
                                layout: layout.as_ref(),
                                entries: binding_entries.as_slice(),
                            })
                        };
                        let idx = self.submit_basic_op(
                            fwd_pipeline.as_ref(),
                            &params,
                            Some(K::FWD_FN_NAME),
                            &cfg
                        );
                        // todo: test that this is safe, then insert new data into rhs
                        // if (rhs_count == 1) {
                        //     self.queue.on_submitted_work_done(|| {
                        //         let rhs_data = rhs.data.clone(); // now 2
                        //         rhs_data.buf.destroy();
                        //     })
                        // }
                        // rhs.data.buf = output;
                        Ok(self.build_tensor(shape, strides, output))
                    } else {
                        lhs.id = unique_id();
                        let params: wgpu::BindGroup = {
                            let binding_entries = to_entries([
                                lhs.data.as_entire_binding(),
                                rhs.data.as_entire_binding(),
                                meta_buffer.as_entire_binding(),
                                output.as_entire_binding(),
                            ]);
                            self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("binary kernel params"),
                                layout: layout.as_ref(),
                                entries: binding_entries.as_slice(),
                            })
                        };
                        let idx = self.submit_basic_op(
                            fwd_pipeline.as_ref(),
                            &params,
                            Some(K::FWD_FN_NAME),
                            &cfg
                        );
                        Ok(self.build_tensor(shape, strides, output))
                    }
                } else {
                    let output = self.create_vec::<E>(meta.numel as usize);
                    let params: wgpu::BindGroup = {
                        let binding_entries = to_entries([
                            lhs.data.as_entire_binding(),
                            rhs.data.as_entire_binding(),
                            meta_buffer.as_entire_binding(),
                            output.as_entire_binding(),
                        ]);
                        self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("binary kernel params"),
                            layout: layout.as_ref(),
                            entries: binding_entries.as_slice(),
                        })
                    };
                    let idx = self.submit_basic_op(
                        fwd_pipeline.as_ref(),
                        &params,
                        Some(K::FWD_FN_NAME),
                        &cfg
                    );
                    Ok(self.build_tensor(shape, strides, output))
                }
            }
            _ => unreachable!(),
        }
    }

    // NOTE: if it becomes possible for grad_out to be broadcasted, (i.e. if #366 is resolved), we
    // need to pass an elems_per_thread argument to the backward cuda kernels, as we do in sum_to.
    fn backward<S: Shape>(
        &self,
        op: K,
        lhs: &impl Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &impl Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        todo!();
        // let bwd_lhs_fn = self
        //     .dev
        //     .get_func(K::MODULE_NAME, K::BWD_LHS_FN_NAME)
        //     .unwrap();

        // let bwd_rhs_fn = self
        //     .dev
        //     .get_func(K::MODULE_NAME, K::BWD_RHS_FN_NAME)
        //     .unwrap();

        // let shape = lhs.shape();
        // let (lhs_strides, lhs_len) = (lhs.strides(), lhs.len());
        // let (rhs_strides, rhs_len) = (rhs.strides(), rhs.len());

        // let numel = shape.num_elements();
        // let cfg = launch_cfg::<128>(numel as u32);

        // let ((out_dims1, out_strides1), rhs_strides1) = permute_for_binary_backward(
        //     shape.concrete(),
        //     shape.strides(),
        //     rhs_strides,
        //     lhs_strides,
        // );

        // let ((out_dims2, out_strides2), lhs_strides2) = permute_for_binary_backward(
        //     shape.concrete(),
        //     shape.strides(),
        //     lhs_strides,
        //     rhs_strides,
        // );

        // let mut info: Vec<usize> = Vec::with_capacity(6 * S::NUM_DIMS);
        // info.extend(out_dims1);
        // info.extend(out_strides1);
        // info.extend(rhs_strides1);
        // info.extend(out_dims2);
        // info.extend(out_strides2);
        // info.extend(lhs_strides2);
        // let info = self.dev.htod_copy(info)?;

        // match (lhs.data(), rhs.data()) {
        //     (Some(lhs_buf), Some(rhs_buf)) => {
        //         let params_lhs = (
        //             op.clone(),      // const OP_STRUCT op,
        //             numel,           // const size_t numel,
        //             S::NUM_DIMS,     // const size_t num_dims,
        //             &info,           // const size_t *info,
        //             lhs_buf,         // const TYPENAME *lhs,
        //             grad_lhs,        // TYPENAME *grad_lhs,
        //             numel / lhs_len, // const size_t chunk_len,
        //             rhs_buf,         // const TYPENAME *rhs,
        //             grad_out,        // const TYPENAME *grad_out
        //         );
        //         let params_rhs = (
        //             op,              // const OP_STRUCT op,
        //             numel,           // const size_t numel,
        //             S::NUM_DIMS,     // const size_t num_dims,
        //             &info,           // const size_T * info,
        //             lhs_buf,         // const TYPENAME *lhs,
        //             rhs_buf,         // const TYPENAME *rhs,
        //             grad_rhs,        // TYPENAME *grad_rhs,
        //             numel / rhs_len, // const size_t chunk_len,
        //             grad_out,        // const TYPENAME *grad_out
        //         );

        //         self.par_stream.wait_for_default()?;
        //         unsafe { bwd_lhs_fn.launch_on_stream(&self.par_stream, cfg, params_lhs) }?;
        //         unsafe { bwd_rhs_fn.launch(cfg, params_rhs) }?;
        //         self.dev.wait_for(&self.par_stream)?;
        //     }
        //     (None, None) => {
        //         let params_lhs = (
        //             op.clone(),      // const OP_STRUCT op,
        //             numel,           // const size_t numel,
        //             S::NUM_DIMS,     // const size_t num_dims,
        //             &info,           // const size_t *info,
        //             0u64,            // const TYPENAME *lhs,
        //             grad_lhs,        // TYPENAME *grad_lhs,
        //             numel / lhs_len, // const size_t chunk_len,
        //             0u64,            // const TYPENAME *rhs,
        //             grad_out,        // const TYPENAME *grad_out
        //         );
        //         let params_rhs = (
        //             op,              // const OP_STRUCT op,
        //             numel,           // const size_t numel,
        //             S::NUM_DIMS,     // const size_t num_dims,
        //             &info,           // const size_T * info,
        //             0u64,            // const TYPENAME *lhs,
        //             0u64,            // const TYPENAME *rhs,
        //             grad_rhs,        // TYPENAME *grad_rhs,
        //             numel / rhs_len, // const size_t chunk_len,
        //             grad_out,        // const TYPENAME *grad_out
        //         );

        //         self.par_stream.wait_for_default()?;
        //         unsafe { bwd_lhs_fn.launch_on_stream(&self.par_stream, cfg, params_lhs) }?;
        //         unsafe { bwd_rhs_fn.launch(cfg, params_rhs) }?;
        //         self.dev.wait_for(&self.par_stream)?;
        //     }
        //     _ => unreachable!(),
        // }

        // Ok(())
    }
}
