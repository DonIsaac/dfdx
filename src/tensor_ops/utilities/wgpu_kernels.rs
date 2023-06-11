use crate::{
    shapes::{Dtype, Shape, Unit},
    tensor::wgpu::{LayoutType, WgpuVec},
    tensor::*,
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};
use ::wgpu::{
    self,
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup,
};
use core::mem::{align_of, size_of};
use std::{borrow::Cow, sync::Arc, vec::Vec};

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
macro_rules! params {
    ($self:expr, $pipeline:expr; $($x:expr),+ $(,)? ) => {
        {
            let entries = to_entries([$($x.as_entire_binding()),+]);
            $self.dev.create_bind_group(&::wgpu::BindGroupDescriptor {
                label: None,
                layout: &($pipeline).get_bind_group_layout(0),
                entries: &entries
            })
        }
    }
}

pub trait BinaryOpWgpuKernel<E: Unit> {
    const HAS_CONST_DF: bool;

    /// Shader source code
    const WGSL_SRC: &'static str;

    /// Unique name for the kernel
    const MODULE_NAME: &'static str;

    /// Name of function in the .wgsl file
    const FWD_FN_NAME: &'static str;

    /// Name of function in the .wgs file
    const BWD_LHS_FN_NAME: &'static str;

    /// Name of function in the .wgsl file
    const BWD_RHS_FN_NAME: &'static str;

    const ALL_FN_NAMES: [&'static str; 3] = [
        Self::FWD_FN_NAME,
        Self::BWD_LHS_FN_NAME,
        Self::BWD_RHS_FN_NAME,
    ];
}

macro_rules! wgpu_binary {
    ($Op:path, $TypeName:ty, $Ptx:tt, $Fwd:tt, $Bwd_Lhs:tt, $Bwd_Rhs:tt) => {
        impl crate::tensor_ops::wgpu_kernels::BinaryOpWgpuKernel<$TypeName> for $Op {
            const HAS_CONST_DF: bool = false;
            const WGSL_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_LHS_FN_NAME: &'static str = $Bwd_Lhs;
            const BWD_RHS_FN_NAME: &'static str = $Bwd_Rhs;
        }
    };
    (const_df() $Op:path, $TypeName:ty, $Ptx:tt, $Fwd:tt, $Bwd_Lhs:tt, $Bwd_Rhs:tt) => {
        impl crate::tensor_ops::wgpu_kernels::BinaryOpWgpuKernel<$TypeName> for $Op {
            const HAS_CONST_DF: bool = true;
            const WGSL_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_LHS_FN_NAME: &'static str = $Bwd_Lhs;
            const BWD_RHS_FN_NAME: &'static str = $Bwd_Rhs;
        }
    };
}
pub(crate) use wgpu_binary;

/*
impl<E: Dtype, K: BinaryOpCudaKernel<E> + DeviceRepr + Clone> BinaryKernel<K, E> for Cuda {
    const BACKWARD_WITHOUT_DATA: bool = K::HAS_CONST_DF;
    fn forward<S: Shape>(
        &self,
        op: K,
        lhs: Cow<Tensor<S, E, Self>>,
        rhs: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        if !self.dev.has_func(K::MODULE_NAME, K::FWD_FN_NAME) {
            self.dev
                .load_ptx(K::PTX_SRC.into(), K::MODULE_NAME, &K::ALL_FN_NAMES)?;
        }
        let fwd_fn = self.dev.get_func(K::MODULE_NAME, K::FWD_FN_NAME).unwrap();

        let shape = match &lhs {
            Cow::Borrowed(lhs) => lhs.shape,
            Cow::Owned(lhs) => lhs.shape,
        };
        let strides = shape.strides();
        let numel = shape.num_elements();
        let cfg = launch_cfg::<128>(numel as u32);

        let lhs_strides = match &lhs {
            Cow::Borrowed(lhs) => lhs.strides,
            Cow::Owned(lhs) => lhs.strides,
        };
        let rhs_strides = match &rhs {
            Cow::Borrowed(rhs) => rhs.strides,
            Cow::Owned(rhs) => rhs.strides,
        };

        let mut info: Vec<usize> = Vec::with_capacity(3 * S::NUM_DIMS);
        info.extend(shape.concrete());
        info.extend(lhs_strides);
        info.extend(rhs_strides);
        let info = self.dev.htod_copy(info)?;

        match (lhs, rhs) {
            (Cow::Borrowed(lhs), Cow::Borrowed(rhs)) => {
                let mut storage = unsafe { self.alloc_empty::<E>(numel) }?;
                let params = (
                    op,
                    numel,             // const size_t numel,
                    S::NUM_DIMS,       // const size_t num_dims,
                    &info,             // const size_t *info,
                    lhs.data.as_ref(), // const float *lhs,
                    rhs.data.as_ref(), // const float *rhs,
                    &mut storage,      // float *out,
                );
                unsafe { fwd_fn.launch(cfg, params) }?;
                Ok(self.build_tensor(shape, strides, storage))
            }
            (Cow::Owned(mut lhs), Cow::Owned(mut rhs)) => {
                let lhs_valid = lhs.strides == lhs.shape.strides();
                let rhs_valid = rhs.strides == rhs.shape.strides();
                if lhs_valid || rhs_valid {
                    let lhs_count = std::sync::Arc::strong_count(&lhs.data);
                    let rhs_count = std::sync::Arc::strong_count(&rhs.data);
                    if rhs_valid && (rhs_count == 1 || !lhs_valid || lhs_count != 1) {
                        rhs.id = unique_id();
                        let params = (
                            op,
                            numel,
                            S::NUM_DIMS,
                            &info,
                            lhs.data.as_ref(),
                            0u64,
                            Arc::make_mut(&mut rhs.data),
                        );
                        unsafe { fwd_fn.launch(cfg, params) }?;
                        Ok(rhs)
                    } else {
                        lhs.id = unique_id();
                        let params = (
                            op,
                            numel,                        // const size_t numel,
                            S::NUM_DIMS,                  // const size_t num_dims,
                            &info,                        // const size_t *info,
                            0u64,                         // const float *lhs,
                            rhs.data.as_ref(),            // const float *rhs,
                            Arc::make_mut(&mut lhs.data), // float *out,
                        );
                        unsafe { fwd_fn.launch(cfg, params) }?;
                        Ok(lhs)
                    }
                } else {
                    let mut storage = unsafe { self.alloc_empty::<E>(numel) }?;
                    let params = (
                        op,
                        numel,             // const size_t numel,
                        S::NUM_DIMS,       // const size_t num_dims,
                        &info,             // const size_t *info,
                        lhs.data.as_ref(), // const float *lhs,
                        rhs.data.as_ref(), // const float *rhs,
                        &mut storage,      // float *out,
                    );
                    unsafe { fwd_fn.launch(cfg, params) }?;
                    Ok(self.build_tensor(shape, strides, storage))
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
        let bwd_lhs_fn = self
            .dev
            .get_func(K::MODULE_NAME, K::BWD_LHS_FN_NAME)
            .unwrap();

        let bwd_rhs_fn = self
            .dev
            .get_func(K::MODULE_NAME, K::BWD_RHS_FN_NAME)
            .unwrap();

        let shape = lhs.shape();
        let (lhs_strides, lhs_len) = (lhs.strides(), lhs.len());
        let (rhs_strides, rhs_len) = (rhs.strides(), rhs.len());

        let numel = shape.num_elements();
        let cfg = launch_cfg::<128>(numel as u32);

        let ((out_dims1, out_strides1), rhs_strides1) = permute_for_binary_backward(
            shape.concrete(),
            shape.strides(),
            rhs_strides,
            lhs_strides,
        );

        let ((out_dims2, out_strides2), lhs_strides2) = permute_for_binary_backward(
            shape.concrete(),
            shape.strides(),
            lhs_strides,
            rhs_strides,
        );

        let mut info: Vec<usize> = Vec::with_capacity(6 * S::NUM_DIMS);
        info.extend(out_dims1);
        info.extend(out_strides1);
        info.extend(rhs_strides1);
        info.extend(out_dims2);
        info.extend(out_strides2);
        info.extend(lhs_strides2);
        let info = self.dev.htod_copy(info)?;

        match (lhs.data(), rhs.data()) {
            (Some(lhs_buf), Some(rhs_buf)) => {
                let params_lhs = (
                    op.clone(),      // const OP_STRUCT op,
                    numel,           // const size_t numel,
                    S::NUM_DIMS,     // const size_t num_dims,
                    &info,           // const size_t *info,
                    lhs_buf,         // const TYPENAME *lhs,
                    grad_lhs,        // TYPENAME *grad_lhs,
                    numel / lhs_len, // const size_t chunk_len,
                    rhs_buf,         // const TYPENAME *rhs,
                    grad_out,        // const TYPENAME *grad_out
                );
                let params_rhs = (
                    op,              // const OP_STRUCT op,
                    numel,           // const size_t numel,
                    S::NUM_DIMS,     // const size_t num_dims,
                    &info,           // const size_T * info,
                    lhs_buf,         // const TYPENAME *lhs,
                    rhs_buf,         // const TYPENAME *rhs,
                    grad_rhs,        // TYPENAME *grad_rhs,
                    numel / rhs_len, // const size_t chunk_len,
                    grad_out,        // const TYPENAME *grad_out
                );

                self.par_stream.wait_for_default()?;
                unsafe { bwd_lhs_fn.launch_on_stream(&self.par_stream, cfg, params_lhs) }?;
                unsafe { bwd_rhs_fn.launch(cfg, params_rhs) }?;
                self.dev.wait_for(&self.par_stream)?;
            }
            (None, None) => {
                let params_lhs = (
                    op.clone(),      // const OP_STRUCT op,
                    numel,           // const size_t numel,
                    S::NUM_DIMS,     // const size_t num_dims,
                    &info,           // const size_t *info,
                    0u64,            // const TYPENAME *lhs,
                    grad_lhs,        // TYPENAME *grad_lhs,
                    numel / lhs_len, // const size_t chunk_len,
                    0u64,            // const TYPENAME *rhs,
                    grad_out,        // const TYPENAME *grad_out
                );
                let params_rhs = (
                    op,              // const OP_STRUCT op,
                    numel,           // const size_t numel,
                    S::NUM_DIMS,     // const size_t num_dims,
                    &info,           // const size_T * info,
                    0u64,            // const TYPENAME *lhs,
                    0u64,            // const TYPENAME *rhs,
                    grad_rhs,        // TYPENAME *grad_rhs,
                    numel / rhs_len, // const size_t chunk_len,
                    grad_out,        // const TYPENAME *grad_out
                );

                self.par_stream.wait_for_default()?;
                unsafe { bwd_lhs_fn.launch_on_stream(&self.par_stream, cfg, params_lhs) }?;
                unsafe { bwd_rhs_fn.launch(cfg, params_rhs) }?;
                self.dev.wait_for(&self.par_stream)?;
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}
 */

/// Metadata passed to kernels. An identical sibling of this struct exists in
/// WGSL for each shader file.
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
static_assertions::const_assert!(align_of::<BinaryKernelMeta<[usize; 2]>>() == 8);
impl<S: Shape> BinaryKernelMeta<S> {
    const CONCRETE_SIZE: usize = size_of::<S::Concrete>();
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
            (size_of::<u32>() * 2) +
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
// impl core::ops::FromResidual<Result<Infallible, PoisonError<RwLockReadGuard<'_, ResourceManager>>>> for WgpuError {
//     fn from_residual(residual: Result<Infallible, PoisonError<RwLockReadGuard<'_, ResourceManager>>>) -> Self {
//         Self::InvalidState("could not lock device because lock was poisoned");
//     }
// }
impl<E: Dtype, K: BinaryOpWgpuKernel<E> + Clone> BinaryKernel<K, E> for Wgpu {
    const BACKWARD_WITHOUT_DATA: bool = K::HAS_CONST_DF;

    fn forward<S: Shape>(
        &self,
        op: K,
        lhs: Cow<Tensor<S, E, Self>>,
        rhs: Cow<Tensor<S, E, Self>>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let pipeline = {
            let resources = self.resources.read().map_err(|_| {
                WgpuError::InvalidState("could not lock device because lock was poisoned")
            })?;
            match resources.get_pipeline::<E>(K::MODULE_NAME, K::FWD_FN_NAME) {
                Some(pipeline) => pipeline,
                None => {
                    drop(resources);
                    let mut resources = self.resources.write().map_err(|_| {
                        WgpuError::InvalidState("could not lock device because lock was poisoned")
                    })?;
                    resources.register_pipeline::<E>(
                        &self.dev.as_ref(),
                        K::MODULE_NAME,
                        K::FWD_FN_NAME,
                        K::WGSL_SRC,
                        LayoutType::Binary,
                    )
                }
            }
        };

        let shape = lhs.shape;
        let strides = shape.strides();
        let meta: BinaryKernelMeta<S> = BinaryKernelMeta::new(&lhs, &rhs);
        let meta_buffer = self.dev.create_buffer_init(&BufferInitDescriptor {
            label: Some("binary kernel meta"),
            contents: &meta.to_bytes(),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // let output: WgpuVec<E> = self.try_alloc_len(meta.numel as usize)?;

        match (lhs, rhs) {
            (Cow::Borrowed(lhs), Cow::Borrowed(rhs)) => {
                let output: WgpuVec<E> = self.try_alloc_len(meta.numel as usize)?;
                let params: BindGroup =
                    params!(self, pipeline; lhs.data, rhs.data, meta_buffer, output);
                let _idx = self.submit(|encoder| {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &params, &[]);
                    pass.dispatch_workgroups(meta.numel, 1, 1);
                });
                return Ok(self.build_tensor(shape, strides, output));
            }
            (Cow::Owned(mut lhs), Cow::Owned(mut rhs)) => {
                let output = self.dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("binary kernel temp output"),
                    size: (meta.numel as u64) * size_of::<E>() as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let lhs_valid = lhs.strides == lhs.shape.strides();
                let rhs_valid = rhs.strides == rhs.shape.strides();
                if lhs_valid || rhs_valid {
                    let lhs_count = std::sync::Arc::strong_count(&lhs.data);
                    let rhs_count = std::sync::Arc::strong_count(&rhs.data);
                    if rhs_valid && (rhs_count == 1 || !lhs_valid || lhs_count != 1) {
                        // output buffer will go into rhs tensor via copy_buffer_to_buffer
                        rhs.id = unique_id();
                        let params: BindGroup =
                            params!(self, pipeline; lhs.data, rhs.data, meta_buffer, output);
                        let _idx = self.submit(|encoder| {
                            {
                                let mut pass = encoder.begin_compute_pass(&Default::default());
                                pass.set_pipeline(&pipeline);
                                pass.set_bind_group(0, &params, &[]);
                                pass.dispatch_workgroups(meta.numel, 1, 1);
                            }
                            debug_assert_eq!(rhs.data.size(), output.size() as usize);
                            encoder.copy_buffer_to_buffer(
                                &output,
                                0,
                                &rhs.data.buffer,
                                0,
                                rhs.data.size() as u64,
                            );
                        });
                        // TODO: enable this and test if it works/makes things faster
                        // self.queue.on_submitted_work_done(|| {
                        //     output.buffer.destroy();
                        // });
                        return Ok(rhs);
                    } else {
                        // output buffer will go into lhs tensor via copy_buffer_to_buffer
                        lhs.id = unique_id();
                        let params: BindGroup =
                            params!(self, pipeline; lhs.data, rhs.data, meta_buffer, output);
                        let _idx = self.submit(|encoder| {
                            {
                                let mut pass = encoder.begin_compute_pass(&Default::default());
                                pass.set_pipeline(&pipeline);
                                pass.set_bind_group(0, &params, &[]);
                                pass.dispatch_workgroups(meta.numel, 1, 1);
                            }
                            debug_assert_eq!(lhs.data.size(), output.size() as usize);
                            encoder.copy_buffer_to_buffer(
                                &output,
                                0,
                                &lhs.data.buffer,
                                0,
                                lhs.data.size() as u64,
                            );
                        });
                        // TODO: enable this and test if it works/makes things faster
                        // self.queue.on_submitted_work_done(|| {
                        //     output.buffer.destroy();
                        // });
                        return Ok(lhs)
                    }
                } else {
                    let output: WgpuVec<E> = self.try_alloc_len(meta.numel as usize)?;
                    let params: BindGroup = params!(self, pipeline; lhs.data, meta_buffer, output);
                    let _idx = self.submit(|encoder| {
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&pipeline);
                        pass.set_bind_group(0, &params, &[]);
                        pass.dispatch_workgroups(meta.numel, 1, 1);
                    });
                    return Ok(self.build_tensor(shape, strides, output));
                }
            }
            _ => unreachable!(),
        }
    }

    fn backward<S: Shape>(
        &self,
        op: K,
        lhs: &impl Tensorlike<S, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &impl Tensorlike<S, E, Self>,
        grad_rhs: &mut Self::Vec,
        grad_out: &Self::Vec,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
