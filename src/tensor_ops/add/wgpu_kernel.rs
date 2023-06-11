use super::{BinaryAddKernelOp as Binary};
use crate::tensor_ops::wgpu_kernels::{wgpu_binary};
#[cfg(feature = "f16")]
use half::f16;

const BADD_SRC: &str = include_str!("./binary_add.wgsl");

wgpu_binary!(
    const_df() Binary,
    f32,
    BADD_SRC,
    "badd_fwd_f32",
    "badd_bwd_lhs_f32",
    "badd_bwd_rhs_f32"
);
#[cfg(feature = "f16")]
wgpu_binary!(
    const_df() Binary,
    f16,
    BADD_SRC,
    "badd_fwd_f16",
    "badd_bwd_lhs_f16",
    "badd_bwd_rhs_f16"
);
wgpu_binary!(
    const_df() Binary,
    u32,
    BADD_SRC,
    "badd_fwd_u32",
    "badd_bwd_lhs_u32",
    "badd_bwd_rhs_u32"
);

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use crate::prelude::*;
    #[cfg(all(feature = "f16", nightly))]
    use half::f16;

    #[test]
    fn test_one_add() {
        let dev: Wgpu = Default::default();
        {
            let a: Tensor<Rank2<2, 2>, f32, _> = dev.ones();
            let b: Tensor<Rank2<2, 2>, f32, _> = dev.ones();
            let expected: Tensor<Rank2<2, 2>, f32, _> = dev.tensor([[2., 2.], [2., 2.]]);
            let c = a + b;
            let c_vec = c.as_vec();
            let expected_vec = expected.as_vec();
            assert_eq!(expected_vec, c_vec);
        }
        #[cfg(all(feature = "f16", nightly))]
        {
            let a: Tensor<Rank2<2, 2>, f16, _> = dev.ones();
            let b: Tensor<Rank2<2, 2>, f16, _> = dev.ones();
            let expected: Tensor<Rank2<2, 2>, f16, _> = dev.tensor([[f16::from_f32(2.), f16::from_f32(2.)], [f16::from_f32(2.), f16::from_f32(2.)]]);
            let c = a + b;
            let c_vec = c.as_vec();
            let expected_vec = expected.as_vec();
            assert_eq!(expected_vec, c_vec);
        }
        {
            let a: Tensor<Rank2<2, 2>, u32, _> = dev.ones();
            let b: Tensor<Rank2<2, 2>, u32, _> = dev.ones();
            let expected: Tensor<Rank2<2, 2>, u32, _> = dev.tensor([[2, 2], [2, 2]]);
            let c = a + b;
            let c_vec = c.as_vec();
            let expected_vec = expected.as_vec();
            assert_eq!(expected_vec, c_vec);
        }

    }
}
