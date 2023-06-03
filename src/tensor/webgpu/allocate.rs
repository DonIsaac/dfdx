use super::{device::*, vec::WgpuVec};
use crate::{
    shapes::*,
    tensor::{
        Tensor,
        masks::triangle_mask,
        storage_traits::*,
        unique_id
    }
};
use std::{sync::Arc, vec::Vec};

impl Wgpu {
    pub(crate) fn build_tensor<S: Shape, E: Unit> (
        &self,
        shape: S,
        strides: S::Concrete,
        data: WgpuVec<E>
    ) -> Tensor<S, E, Self> {
        Tensor {
            id: unique_id(),
            shape,
            strides,
            data: Arc::new(data),
            device: self.clone(),
            tape: Default::default()
        }
    }

    /// Creates a new, unmapped vector of the given length.
    pub(crate) fn create_vec<E>(&self, len: usize) -> WgpuVec<E> {
        WgpuVec::new(self.dev.clone(), len, false)
    }
}

impl<E: Unit> ZerosTensor<E> for Wgpu {
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        // let v = vec![E::ZERO; shape.num_elements()];
        // let gpu_vec: WgpuVec<E> = WgpuVec::from_vec(self.dev.clone(), &v);
        let mut vec: WgpuVec<E> = WgpuVec::new(self.dev.clone(), shape.num_elements(), true);
        vec.copy_fill(E::ZERO);
        vec.unmap();
        Ok(self.build_tensor(shape, strides, vec))
    }
}

impl <E: Unit> ZeroFillStorage<E> for Wgpu {
    fn try_fill_with_zeros(&self, storage: &mut Self::Vec) -> Result<(), Self::Err> {
        storage.copy_fill(E::ZERO);
        Ok(())
    }
}

impl <E: Unit> OnesTensor<E> for Wgpu {
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        // let v = vec![E::ZERO; shape.num_elements()];
        // let gpu_vec: WgpuVec<E> = WgpuVec::from_vec(self.dev.clone(), &v);
        let mut vec: WgpuVec<E> = WgpuVec::new(self.dev.clone(), shape.num_elements(), true);
        vec.copy_fill(E::ONE);
        vec.unmap();
        Ok(self.build_tensor(shape, strides, vec))
    }
}

impl <E: Unit> OneFillStorage<E> for Wgpu {
    fn try_fill_with_ones(&self, storage: &mut Self::Vec) -> Result<(), Self::Err> {
        storage.copy_fill(E::ONE);
        Ok(())
    }
}

impl<E: Unit> TriangleTensor<E> for Wgpu {
    fn try_upper_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let mut data = std::vec![val; shape.num_elements()];
        let offset = diagonal.into().unwrap_or(0);
        triangle_mask(&mut data, &shape, true, offset);
        let vec = WgpuVec::from_vec(self.dev.clone(), &data);
        Ok(self.build_tensor(shape, shape.strides(), vec))
    }

    fn try_lower_tri_like<S: HasShape>(
        &self,
        src: &S,
        val: E,
        diagonal: impl Into<Option<isize>>,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let mut data = std::vec![val; shape.num_elements()];
        let offset = diagonal.into().unwrap_or(0);
        triangle_mask(&mut data, &shape, false, offset);
        let vec = WgpuVec::from_vec(self.dev.clone(), &data);
        Ok(self.build_tensor(shape, shape.strides(), vec))
    }
}
