mod device;
mod vec;
mod allocate;
// mod cache;
pub(crate) mod resources;

pub use device::{Wgpu, WgpuError};
pub(crate) use device::OpLayoutType;
pub(crate) use vec::WgpuVec;
pub(crate) use resources::{WorkGroupSize};

#[cfg(all(test, feature = "wgpu"))]
mod test {
    use super::*;
    use crate::{shapes::*, tensor::*};

    #[test]
    fn test_new_zero_tensor() {
        let dev: Wgpu = Default::default();
        let _a: Tensor<Rank3<2,3,4>, f32, _> = dev.zeros();
    }
}
