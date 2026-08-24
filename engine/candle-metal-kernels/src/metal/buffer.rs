use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{MTLBuffer, MTLResource};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

pub type MetalResource = ProtocolObject<dyn MTLResource>;
pub type MTLResourceOptions = objc2_metal::MTLResourceOptions;

#[derive(Debug)]
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    last_used_command_buffer_id: Arc<AtomicU64>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Buffer {
        Buffer {
            raw,
            last_used_command_buffer_id: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn contents(&self) -> *mut u8 {
        self.data()
    }

    pub fn data(&self) -> *mut u8 {
        use objc2_metal::MTLBuffer as _;
        self.as_ref().contents().as_ptr() as *mut u8
    }

    /// Get the raw pointer to the underlying Metal buffer object.
    /// Used for dependency tracking in the compute encoder.
    pub(crate) fn raw_ptr(&self) -> *const ProtocolObject<dyn MTLBuffer> {
        Retained::as_ptr(&self.raw)
    }

    pub fn length(&self) -> usize {
        self.as_ref().length()
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.as_ref().didModifyRange(range);
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }

    pub fn mark_used_in_command_buffer(&self, command_buffer_id: u64) {
        self.last_used_command_buffer_id
            .store(command_buffer_id, Ordering::Release);
    }

    pub fn last_used_command_buffer_id(&self) -> u64 {
        self.last_used_command_buffer_id.load(Ordering::Acquire)
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            last_used_command_buffer_id: Arc::clone(&self.last_used_command_buffer_id),
        }
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}

impl<'a> From<&'a Buffer> for &'a MetalResource {
    fn from(val: &'a Buffer) -> Self {
        ProtocolObject::from_ref(val.as_ref())
    }
}

pub type BufferMap = HashMap<usize, Vec<Arc<Buffer>>>;
