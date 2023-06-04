extern crate alloc;
use alloc::sync::Arc;
use wgpu::{Buffer};
use core::{marker::PhantomData, any::TypeId};

use super::Wgpu;

#[derive(Debug)]
pub struct WgpuVec<E: 'static> {
    pub(crate) ty: TypeId,
    dev: Arc<Wgpu>,
    pub(crate) buffer: wgpu::Buffer,
    pd: PhantomData<E>
}

// /// Identical to [`wgpu::BufferDescriptor`] except that it uses `len` instead of
// /// size, instead deriving the size from the type of the buffer.
// pub struct WgpuVecDescriptor {
//     pub label: Option<&'static str>,
//     pub usage: wgpu::BufferUsage,
//     pub len: usize,
//     pub mapped_at_creation: bool
// }
// 
// impl <E: 'static> WgpuVec<E> {
//     pub fn new(desc: &WgpuVecDescriptor) {
//         todo!()
//     }
// }
impl<E: 'static> WgpuVec<E> {
    pub fn from_buffer(dev: Arc<Wgpu>, buffer: Buffer) -> Self {
        Self {
            dev,
            buffer,
            ty: TypeId::of::<E>(),
            pd: PhantomData
        }
    }
    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        (self.buffer.size() as usize) / std::mem::size_of::<E>()
    }
}

impl <E: 'static> Into<Buffer> for WgpuVec<E> {
    fn into(self) -> Buffer {
        self.buffer
    }
}

impl<E: 'static> Clone for WgpuVec<E> {
    fn clone(&self) -> Self {
        let new_buffer = match self.buffer.usage() {
        wgpu::BufferUsages::COPY_SRC => {
            todo!()
        },
        _ => panic!("cannot clone buffer with usage {:?}", self.buffer.usage())
       };
       todo!() 
    }
}
