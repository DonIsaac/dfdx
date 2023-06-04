use super::{BinaryAddKernelOp as Binary};
use crate::tensor_ops::wgpu_kernels::{wgpu_binary};

const BADD_SRC: &'static str = include_str!("./binary_add.wgsl");

wgpu_binary!(
    const_df() Binary,
    f32,
    BADD_SRC,
    "badd_fwd_f32",
    "badd_bwd_lhs_f32",
    "badd_bwd_rhs_f32"
);

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use crate::prelude::*;
    #[test]
    fn test_zero_add() {
        let dev: Wgpu = Default::default();
        let a: Tensor<Rank2<2, 2>, f32, _> = dev.ones();
        let b: Tensor<Rank2<2, 2>, f32, _> = dev.ones();
        let expected: Tensor<Rank2<2, 2>, f32, _> = dev.tensor([[2., 2.], [2., 2.]]);
        // {

        //     dbg!(a);
        //     dbg!(b);
        //     dbg!(expected);
        // }
        let c = dbg!(a) + dbg!(b);
        let c_vec = c.as_vec();
        let expected_vec = dbg!(expected).as_vec();
        assert_eq!(expected_vec, c_vec);

    }
}
