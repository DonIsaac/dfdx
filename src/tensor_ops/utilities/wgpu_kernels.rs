use crate::{
    shapes::{Dtype, Shape, Unit},
    tensor::*,
    tensor_ops::ops::{BinaryKernel, UnaryKernel},
};
use std::{borrow::Cow, sync::Arc, vec::Vec};

macro_rules! include_generic_wgsl {
    ($Wgsl:tt, $TypeName:ty) => {
        include_str!($Wgsl)
    };
}

const TYPENAME_TEMPLATE: &'static str = "__TYPENAME__";

pub trait BinaryOpWgpuKernel<E: Unit> {
    const HAS_CONST_DF: bool;

    /// Shader source code
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
    ($Op:path, $TypeName:ty, $Ptx:tt, $Fwd:tt, $Bwd_Lhs:tt, $Bwd_Rhs:tt) => {
        impl crate::tensor_ops::cuda_kernels::BinaryOpWgpuKernel<$TypeName> for $Op {
            const HAS_CONST_DF: bool = false;
            const WGSL_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_LHS_FN_NAME: &'static str = $Bwd_Lhs;
            const BWD_RHS_FN_NAME: &'static str = $Bwd_Rhs;
        }
    };
    (const_df() $Op:path, $TypeName:ty, $Ptx:tt, $Fwd:tt, $Bwd_Lhs:tt, $Bwd_Rhs:tt) => {
        impl crate::tensor_ops::cuda_kernels::BinaryOpWgpuKernel<$TypeName> for $Op {
            const HAS_CONST_DF: bool = true;
            const WGSL_SRC: &'static str = $Ptx;
            const MODULE_NAME: &'static str = $Fwd;
            const FWD_FN_NAME: &'static str = $Fwd;
            const BWD_LHS_FN_NAME: &'static str = $Bwd_Lhs;
            const BWD_RHS_FN_NAME: &'static str = $Bwd_Rhs;
        }
    };
}
