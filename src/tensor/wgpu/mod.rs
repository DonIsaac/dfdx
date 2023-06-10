mod device;
mod vec;
mod allocate;
mod resources;

pub use vec::WgpuVec;
pub use device::{Wgpu, WgpuError};

#[cfg(test)]
mod tests{
    use super::*;
    use crate::{shapes::*, tensor::*};

    #[test]
    fn test_copy_ones() {
        let dev: Wgpu = Default::default();
        let mut zero: Tensor<Rank2<2,2>, f32, _> = dev.zeros();
        let one: Tensor<Rank2<2, 2>, f32, _> = dev.ones();
        zero.fill_with_ones();

        let zero_vec = zero.as_vec();
        let one_vec = one.as_vec();
        assert_eq!(zero_vec, one_vec);
    }

    #[test]
    fn test_tri_upper() {
        let dev: Wgpu = Default::default();
        let x: f32 = Default::default();
        let o: f32 = 0.0;

        let triu: Tensor<Rank2<3, 3>, f32, _> = dev.upper_tri(x, None);
        let triu_manual: Tensor<Rank2<3, 3>, f32, _> = dev.tensor([
            [x, x, x],
            [o, x, x],
            [o, o, x]
        ]) ;

        assert_eq!(triu.as_vec(), triu_manual.as_vec());
    }

    #[test]
    fn test_tri_lower() {
        let dev: Wgpu = Default::default();
        let x: f32 = Default::default();
        let o: f32 = 0.0;

        let triu: Tensor<Rank2<3, 3>, f32, _> = dev.lower_tri(x, None);
        let triu_manual: Tensor<Rank2<3, 3>, f32, _> = dev.tensor([
            [x, o, o],
            [x, x, o],
            [x, x, x]
        ]) ;

        assert_eq!(triu.as_vec(), triu_manual.as_vec());
    }
}
