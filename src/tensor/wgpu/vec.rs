extern crate alloc;
use alloc::sync::Arc;
use core::{any::TypeId, marker::PhantomData};
use wgpu::{util::DeviceExt, Buffer, BufferDescriptor, BufferUsages};

use super::Wgpu;

#[derive(Debug)]
pub struct WgpuVec<E: 'static> {
    pub(crate) ty: TypeId,
    dev: Arc<Wgpu>,
    pub(crate) buffer: Buffer,
    pd: PhantomData<E>,
}

impl<E: 'static> WgpuVec<E> {
    pub fn from_buffer(dev: Arc<Wgpu>, buffer: Buffer) -> Self {
        Self {
            dev,
            buffer,
            ty: TypeId::of::<E>(),
            pd: PhantomData,
        }
    }
    // V is convertable to a slice with as_slice()
    pub fn from_vec<V>(dev: Arc<Wgpu>, vec: &V) -> Self
    where
        V: AsRef<[E]>,
    {
        let contents = unsafe {
            let (prefix, slice, suffix) = vec.as_ref().align_to::<u8>();
            assert_eq!(prefix.len(), 0);
            assert_eq!(suffix.len(), 0);
            slice
        };
        let buffer = dev
            .dev
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            });
        Self::from_buffer(dev, buffer)
    }

    /// Creates a new general-purpose buffer, appropriate for most
    /// tensor storage uses.
    pub fn storage(dev: Arc<Wgpu>, len: usize) -> Self {
        let buffer = dev.dev.create_buffer(&BufferDescriptor {
            label: None,
            size: (len * std::mem::size_of::<E>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self::from_buffer(dev, buffer)
    }

    pub fn storage_mut(dev: Arc<Wgpu>, len: usize) -> Self {
        let buffer = dev.dev.create_buffer(&BufferDescriptor {
            label: None,
            size: (len * std::mem::size_of::<E>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self::from_buffer(dev, buffer)
    }

    /// Returns the number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        (self.buffer.size() as usize) / std::mem::size_of::<E>()
    }

    /// Size, in bytes, of the underlying GPU buffer.
    #[inline]
    pub fn size(&self) -> usize {
        self.buffer.size() as usize
    }
}

impl<E: 'static> Into<Buffer> for WgpuVec<E> {
    fn into(self) -> Buffer {
        self.buffer
    }
}

impl<E: 'static + Copy> Into<Vec<E>> for WgpuVec<E> {
    fn into(self) -> Vec<E> {
        debug_assert_eq!(TypeId::of::<E>(), self.ty);
        let usages = self.buffer.usage();

        match usages {
            // This buffer is mappable and doesn't need to be copied into a
            // mappable buffer. Usually only happens when
            // MAPPABLE_PRIMARY_BUFFERS is enabled.
            BufferUsages::MAP_READ => {
                // Panics if buffer isn't immediately mappable
                let view = self.buffer.slice(..).get_mapped_range();
                let mut v: Vec<E> = Vec::with_capacity(self.buffer.size() as usize);
                unsafe {
                    let (prefix, slice, suffix) = view.align_to::<E>();
                    assert_eq!(prefix.len(), 0);
                    assert_eq!(suffix.len(), 0);
                    v.copy_from_slice(slice);
                }
                return v;
            }

            // buffer is copyable on the gpu, so we copy it into a mappable
            // buffer and move it over
            BufferUsages::COPY_SRC => {
                let new_buffer = self.dev.dev.create_buffer(&BufferDescriptor {
                    label: None,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    size: self.buffer.size(),
                    mapped_at_creation: false,
                });
                let copy_id = self.dev.submit(|encoder| {
                    encoder.copy_buffer_to_buffer(
                        &self.buffer,
                        0,
                        &new_buffer,
                        0,
                        self.buffer.size(),
                    )
                });
                new_buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |res| res.unwrap());
                self.dev
                    .dev
                    .poll(wgpu::Maintain::WaitForSubmissionIndex(copy_id));

                let view = new_buffer.slice(..).get_mapped_range();
                let mut v: Vec<E> = Vec::with_capacity(new_buffer.size() as usize);
                unsafe {
                    let (prefix, slice, suffix) = view.align_to::<E>();
                    assert_eq!(prefix.len(), 0);
                    assert_eq!(suffix.len(), 0);
                    v.copy_from_slice(slice);
                }
                return v;
            }
            _ => {
                panic!("cannot map wgpu buffer with usages: {:?}", usages);
            }
        }
    }
}

impl<E: 'static> Clone for WgpuVec<E> {
    fn clone(&self) -> Self {
        let usages = self.buffer.usage();

        // Type of variable the buffer can be used as. Other usages are added later.
        let memory_layout_type = match usages {
            wgpu::BufferUsages::STORAGE => wgpu::BufferUsages::STORAGE,
            wgpu::BufferUsages::UNIFORM => wgpu::BufferUsages::UNIFORM,
            wgpu::BufferUsages::INDEX => wgpu::BufferUsages::INDEX,
            wgpu::BufferUsages::VERTEX => wgpu::BufferUsages::VERTEX,
            _ => panic!("cannot clone buffer with usage {:?}", usages),
        };

        // Allocate a new buffer on the GPU
        let new_buffer = match usages {
            wgpu::BufferUsages::COPY_SRC => {
                let buf = self.dev.dev.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: self.buffer.size(),
                    usage: memory_layout_type
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                self.dev.submit(|encoder| {
                    encoder.copy_buffer_to_buffer(&self.buffer, 0, &buf, 0, self.buffer.size())
                });
                buf
            }
            // todo: handle map_read
            _ => panic!("cannot clone buffer with usage {:?}", usages),
        };

        return Self::from_buffer(self.dev.clone(), new_buffer);
    }
}
