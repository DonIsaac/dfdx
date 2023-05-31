use core::marker::PhantomData;
use core::ops::Deref;
use core::{any::TypeId};
use wgpu::{self, util::DeviceExt, Buffer, Device};
use std::sync::Arc;

#[derive(Debug)]
pub struct WgpuVec<E> {
    pub(crate) dev: Arc<Device>,
    pub(crate) buf: Buffer,
    pub(crate) ty: TypeId,
    pub(crate) is_mapped: bool,
    pty: PhantomData<E>,
}

unsafe impl<T: Send> Send for WgpuVec<T> {}
unsafe impl<T: Sync> Sync for WgpuVec<T> {}

impl<E: 'static> WgpuVec<E> {
    pub(crate) fn new(dev: Arc<Device>, len: usize, mapped_at_creation: bool) -> Self {
        let buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None, // todo
            size: (std::mem::size_of::<E>() * len) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation,
        });
        Self {
            dev,
            buf,
            ty: TypeId::of::<E>(),
            is_mapped: mapped_at_creation,
            pty: Default::default(),
        }
    }

    // pub(crate) fn map(&self) {
    //     if !self.is_mapped {
    //         self.buf
    //             .slice(..)
    //             .map_async(wgpu::MapMode::Read, |res| res.unwrap());
    //         self.is_mapped = true;
    //     }
    // }
    pub(crate) fn map(&mut self) {
        if !self.is_mapped {
            self.buf
                .slice(..)
                .map_async(wgpu::MapMode::Write, |res| res.unwrap());
            self.is_mapped = true;
        }
    }
    pub(crate) fn unmap(&self) {
        self.buf.unmap()
    }

    /// Returns the number of elements in the vec.
    pub(crate) fn len(&self) -> usize {
        self.buf.size() as usize / std::mem::size_of::<E>()
    }

    /// Returns the number of bytes used by the vec's elements.
    /// 
    /// This is different than the size of the vec itself. This function doesn't
    /// account for overhead from the WgpuVec struct itself.
    pub(crate) fn size(&self) -> usize {
        self.buf.size() as usize
    }

    pub(crate) fn view(&self) -> wgpu::BufferView {
        self.buf.slice(..).get_mapped_range()
    }
    pub(crate) fn view_mut(&self) -> wgpu::BufferViewMut {
        self.buf.slice(..).get_mapped_range_mut()
    }

    /* 
    /// Panics if the buffer isn't mapped
    pub(crate) fn as_slice<'s>(&'s self) -> &'s [E] {
        let slice: wgpu::BufferSlice<'s> = self.buf.slice(..);
        let view: wgpu::BufferView<'s> = slice.get_mapped_range();

        // # Safety
        // 1. This is effectively a transmute. Layouts are guaranteed to be the
        //    same because of the guarantees provided by the constructor.
        //    However, just to be doubly safe, we check the type id of both E
        //    and the type of the buffer during development
        // 2. The slice is actually constructed by copying a *mut pointer into a
        //    slice. This means mutations to the result of this function will
        //    affect this buffer. This is fine because the returned slice is
        //    immutable, so invalid mutations will get caught by the compiler.
        debug_assert_eq!(TypeId::of::<E>(), self.ty, "WgpuBuffer::as_slice(): type mismatch between buffer's contents and expected type");
        unsafe {
            // let (prefix, data, suffix) = view.align_to::<E>();
            // assert_eq!(prefix.len(), 0);
            // assert_eq!(suffix.len(), 0);
            // return data
            let aligned = view.align_to::<E>();
            let data: &'s [E] = aligned.1;
            return data;
        }
        // std::slice::from_raw_parts(slice.get_mapped_range().as_ptr() as *const E, self.len())
    }
    */
}

// impl <E : Clone + 'static> WgpuVec<E> {
//     pub(crate) fn as_vec(&self) -> Vec<E> {
//         self.as_slice().to_vec()
//     }
// }
impl<E> Clone for WgpuVec<E> {
    // todo: clone when on gpu
    fn clone(&self) -> Self {
        if self.is_mapped {
            todo!("clone for mapped buffers not yet supported")
        } else {
            let copied_data = self.buf.slice(..).get_mapped_range();
            // let foo = *copied_data;
            let new_buf = self
                .dev
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: copied_data.deref(),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            Self {
                buf: new_buf,
                is_mapped: false, // buffers created by create_buffer_init get unmapped after initialization
                dev: self.dev.clone(),
                ..*self
            }
        }
    }
}
impl<E: Clone> Into<Vec<E>> for WgpuVec<E> {
    fn into(self) -> Vec<E> {
        let slice = self.buf.slice(..);

        if !self.is_mapped {
            slice.map_async(wgpu::MapMode::Read, |res| res.unwrap());
            self.dev.poll(wgpu::Maintain::Wait);
        }

        // note(5/31/23): get_mapped_range() allocs on web and doesn't on
        // native. as_ref() does not mutate either. These are impl details and
        // cannot be relied on. Unfortunately there is no way to perform a
        // zero-copy move into a vec.
        let view = slice.get_mapped_range();
        // # Safety
        // Safety considerations here are the same as from Clone, with some
        // slight differences. This method consumes self, so ownership may be
        // safely passed into a vec without a move, but this is difficult to
        // perform. So we copy anyways.
        let data: &[E] = unsafe {
            let (prefix, data, suffix) = view.align_to::<E>();
            assert_eq!(prefix.len(), 0);
            assert_eq!(suffix.len(), 0);
            data
        };

        let vec: Vec<E> = data.to_vec(); // copy is here
        vec
    }
}
