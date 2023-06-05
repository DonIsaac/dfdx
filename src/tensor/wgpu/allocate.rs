use super::device::{Wgpu, WgpuError};
use super::vec::WgpuVec;
use crate::{
    shapes::*,
    tensor::{masks::triangle_mask, storage_traits::*, unique_id, Cpu, Tensor},
};
use rand::Rng;
use std::sync::Arc;

impl Wgpu {
    pub(crate) fn build_tensor<S: Shape, E: Unit>(
        &self,
        shape: S,
        strides: S::Concrete,
        data: WgpuVec<E>,
    ) -> Tensor<S, E, Self> {
        Tensor {
            id: unique_id(),
            data: Arc::new(data),
            shape,
            strides,
            device: self.clone(),
            tape: Default::default(),
        }
    }

    pub(crate) fn tensor_from_vec<S: Shape, E: Unit>(
        &self,
        shape: S,
        vec: &Vec<E>,
    ) -> Tensor<S, E, Self> {
        let data: WgpuVec<E> = WgpuVec::from_vec(Arc::new(self.clone()), vec);
        let strides = shape.strides();
        self.build_tensor(shape, strides, data)
    }
}

impl<E: Unit> ZerosTensor<E> for Wgpu {
    fn try_zeros_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        let numel = shape.num_elements();
        let mut storage: Self::Vec = WgpuVec::storage_mut(Arc::new(self.clone()), numel);
        self.try_fill_with_zeros(&mut storage)?;
        Ok(self.build_tensor(shape, strides, storage))
    }
}

impl<E: Unit> ZeroFillStorage<E> for Wgpu {
    fn try_fill_with_zeros(&self, storage: &mut Self::Vec) -> Result<(), Self::Err> {
        if !storage
            .buffer
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST)
        {
            return Err(WgpuError::InvalidBufferUsage(
                "cannot clear a buffer without COPY_DST".to_string(),
            ));
        }

        // panics if storage is mapped
        self.submit(|encoder| {
            encoder.clear_buffer(&storage.buffer, 0, None);
        });
        Ok(())
    }
}

impl<E: Unit> OnesTensor<E> for Wgpu {
    fn try_ones_like<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let strides = shape.strides();
        let numel = shape.num_elements();
        let v = vec![E::ONE; numel];
        let storage: WgpuVec<E> = WgpuVec::from_vec(Arc::new(self.clone()), &v);
        Ok(self.build_tensor(shape, strides, storage))
    }
}

impl<E: Unit> OneFillStorage<E> for Wgpu {
    fn try_fill_with_ones(&self, storage: &mut Self::Vec) -> Result<(), Self::Err> {
        if !storage
            .buffer
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST)
        {
            return Err(WgpuError::InvalidBufferUsage(
                "cannot clear a buffer without COPY_DST".to_string(),
            ));
        }
        let cpu_ones = vec![E::ONE; storage.len()];
        let raw_ones: &[u8] = unsafe {
            let (prefix, slice, suffix) = cpu_ones.as_slice().align_to::<u8>();
            assert_eq!(prefix.len(), 0);
            assert_eq!(suffix.len(), 0);
            slice
        };
        Ok(self.queue.write_buffer(&storage.buffer, 0, raw_ones))
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
        Ok(self.tensor_from_vec(shape, &data))
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
        Ok(self.tensor_from_vec(shape, &data))
    }
}

impl<E: Unit> SampleTensor<E> for Wgpu
where
    Cpu: SampleTensor<E>,
{
    fn try_sample_like<S: HasShape, D: rand_distr::Distribution<E>>(
        &self,
        src: &S,
        distr: D,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let mut buf = Vec::with_capacity(shape.num_elements());
        {
            #[cfg(not(feature = "no-std"))]
            let mut rng = self.cpu.rng.lock().unwrap();
            #[cfg(feature = "no-std")]
            let mut rng = self.cpu.rng.lock();
            buf.resize_with(shape.num_elements(), || rng.sample(&distr));
        }
        Ok(self.tensor_from_vec(shape, &buf))
    }

    fn try_fill_with_distr<D: rand_distr::Distribution<E>>(
        &self,
        storage: &mut Self::Vec,
        distr: D,
    ) -> Result<(), Self::Err> {
        if !storage
            .buffer
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST)
        {
            return Err(WgpuError::InvalidBufferUsage(
                "cannot fill a buffer without COPY_DST".to_string(),
            ));
        }

        let mut buf = Vec::with_capacity(storage.len());
        {
            #[cfg(not(feature = "no-std"))]
            let mut rng = self.cpu.rng.lock().unwrap();
            #[cfg(feature = "no-std")]
            let mut rng = self.cpu.rng.lock();
            buf.resize_with(storage.len(), || rng.sample(&distr));
        }

        let bytes = unsafe {
            let (prefix, slice, suffix) = buf.as_slice().align_to::<u8>();
            assert_eq!(prefix.len(), 0);
            assert_eq!(suffix.len(), 0);
            slice
        };

        self.queue.write_buffer(&storage.buffer, 0, bytes);
        Ok(())
    }
}

impl<E: Unit> TensorFromVec<E> for Wgpu {
    fn try_tensor_from_vec<S: Shape>(
        &self,
        src: Vec<E>,
        shape: S,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        Ok(self.tensor_from_vec(shape, &src))
    }
}

// impl<S: Shape, E: Unit> TensorToArray<S, E> for Wgpu {
//     type Array;

//     fn tensor_to_array<T>(&self, tensor: &Tensor<S, E, Self, T>) -> Self::Array {
//         todo!()
//     }
// }
