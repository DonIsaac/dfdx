use core::any::TypeId;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ops::Deref;
use core::slice;
use std::sync::Arc;
use wgpu::{self, util::DeviceExt, Buffer, Device};

#[derive(Debug)]
pub struct WgpuVec<E> {
    pub(crate) dev: Arc<Device>,
    /// The data type of the elements in the buffer. This reflects `E` in
    /// `WgpuVec<E>`. This info is needed at runtime to determine what shaders
    /// to use when dispatching operations.
    pub(crate) ty: TypeId,
    pub(crate) buf: Buffer,
    /// Whether the buffer is mapped.
    /// 
    /// Mapped buffers are able to me read/modified (depending on how they were
    /// mapped) by the host device. They must be unmapped before they can be
    /// used by the GPU.
    pub(crate) is_mapped: bool,
    /// `E` is needed at compile time for operator overloading, but is not
    /// needed by data in this struct. PhantomData is used to make the compiler
    /// happy.
    pty: PhantomData<E>,
}
static_assertions::assert_impl_all!(WgpuVec<u8>: Send, Sync);

impl<E: 'static> WgpuVec<E> {
    /// Internal constructor with full control over the buffer. Overloaded by
    /// other, public constructors.
    pub(crate) fn new(
        dev: Arc<Device>,
        label: wgpu::Label,
        len: usize,
        usage: wgpu::BufferUsages,
        mapped_at_creation: bool
    ) -> Self {
        let buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: (std::mem::size_of::<E>() * len) as u64,
            usage,
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

    pub(crate) fn storage(dev: Arc<Device>, len: usize, mapped_at_creation: bool) -> Self {
        Self::new(
            dev,
            None,
            len,
            wgpu::BufferUsages::STORAGE,
            mapped_at_creation,
        )
    }

    pub(crate) fn uniform(dev: Arc<Device>, len: usize, mapped_at_creation: bool) -> Self {
        Self::new(
            dev,
            None,
            len,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation
        )
    }

    pub fn from_vec(dev: Arc<Device>, vec: &Vec<E>) -> Self {
        let raw_bytes: &[u8] = if vec.len() == 0 {
            &[]
        } else {
            unsafe {
                let (prefix, data, suffix) = vec.as_slice().align_to::<u8>();
                assert_eq!(prefix.len(), 0);
                assert_eq!(suffix.len(), 0);
                data
            }
        };
        let buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: raw_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        return Self {
            dev,
            buf,
            ty: TypeId::of::<E>(),
            is_mapped: false,
            pty: Default::default(),
        };
    }

    pub(crate) fn map(&mut self) {
        if !self.is_mapped {
            self.buf
                .slice(..)
                .map_async(wgpu::MapMode::Write, |res| res.unwrap());
            self.is_mapped = true;
        }
    }
    /// Flushes any pending write operations and unmaps the buffer from host
    /// memory. Or, more simply, moves the buffer to the GPU.
    pub(crate) fn unmap(&mut self) {
        self.buf.unmap();
        self.is_mapped = false;
    }

    /// Returns the number of elements in the vec.
    pub fn len(&self) -> usize {
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

    pub(crate) fn as_entire_binding(&self) -> wgpu::BindingResource {
        self.buf.as_entire_binding()
    }

    pub(crate) fn as_entire_buffer_binding(&self) -> wgpu::BufferBinding {
        self.buf.as_entire_buffer_binding()
    }
}

impl<'a, E> Into<&'a wgpu::Buffer> for &'a WgpuVec<E> {
    fn into(self) -> &'a wgpu::Buffer {
        &self.buf
    }
}

impl<E> Into<wgpu::Buffer> for WgpuVec<E> {
    fn into(self) -> wgpu::Buffer {
        self.buf
    }
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

impl<E: 'static + Copy> WgpuVec<E> {
    /// Copies the contents of the given slice into the buffer.
    ///
    /// # Panics
    /// 1. If the buffer is not mapped
    /// 2. If the length of the given slice does not match the length of the buffer
    /// 3. If alignment is incorrect (should never happen)
    pub fn copy_into(&mut self, data: &[E]) {
        assert_eq!(self.len(), data.len());
        debug_assert_eq!(
            self.ty,
            TypeId::of::<E>(),
            "WgpuBuffer::copy_into(): type mismatch between buffer's contents and expected type"
        );
        let mut view = self.view_mut();
        unsafe {
            let (prefix, raw, suffix) = data.align_to::<u8>();
            assert_eq!(prefix.len(), 0);
            assert_eq!(suffix.len(), 0);
            view.copy_from_slice(raw);
        }
    }

    pub fn copy_fill(&mut self, data: E) {
        if self.len() == 0 {
            return;
        }
        debug_assert!(self.len() % size_of::<E>() == 0);
        debug_assert_eq!(
            self.ty,
            TypeId::of::<E>(),
            "WgpuBuffer::copy_fill(): type mismatch between buffer's contents and expected type"
        );

        let slice = slice::from_ref(&data);
        let mut view = self.view_mut();

        unsafe {
            let (p, raw, s) = slice.align_to::<u8>();
            assert_eq!(p.len(), 0);
            assert_eq!(s.len(), 0);
            // view.as_chunks_mut::<size_of::<E>()>();
            for chunk in view.chunks_exact_mut(size_of::<E>()) {
                chunk.copy_from_slice(raw);
            }
            // self.view_mut().
        }
    }
}

impl<E: 'static + Clone> WgpuVec<E> {
    /// Clones the contents of the given slice into the buffer.
    ///
    /// # Panics
    /// 1. If the buffer is not mapped
    /// 2. If the length of the given slice does not match the length of the buffer
    /// 3. If alignment is incorrect (should never happen)
    pub fn clone_into(&mut self, data: &[E]) {
        assert_eq!(self.len(), data.len());
        debug_assert_eq!(
            self.ty,
            TypeId::of::<E>(),
            "WgpuBuffer:clone_into(): type mismatch between buffer's contents and expected type"
        );
        let mut view = self.view_mut();
        unsafe {
            let (prefix, raw, suffix) = data.align_to::<u8>();
            assert_eq!(prefix.len(), 0);
            assert_eq!(suffix.len(), 0);
            view.copy_from_slice(raw);
        }
    }

    pub fn clone_fill(&mut self, data: &[E]) {
        if self.len() == 0 {
            return;
        }
        debug_assert!(self.len() % size_of::<E>() == 0);
        debug_assert_eq!(
            self.ty,
            TypeId::of::<E>(),
            "WgpuBuffer:clone_fill(): type mismatch between buffer's contents and expected type"
        );

        let mut view = self.view_mut();

        unsafe {
            let (p, raw, s) = data.align_to::<u8>();
            assert_eq!(p.len(), 0);
            assert_eq!(s.len(), 0);
            // view.as_chunks_mut::<size_of::<E>()>();
            for chunk in view.chunks_exact_mut(size_of::<E>()) {
                chunk.copy_from_slice(raw);
            }
            // self.view_mut().
        }
        todo!()
    }
}
