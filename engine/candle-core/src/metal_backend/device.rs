use crate::{DType, Result};

#[cfg(feature = "ug")]
use candle_metal_kernels::metal::ComputePipeline;
use candle_metal_kernels::{
    metal::{
        BlitCommandsGuard, Buffer, BufferMap, Commands, CommandsGuard, Device, MTLResourceOptions,
        ResidencySet, ScratchArena, UnifiedScratchArena,
    },
    Kernels,
};
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

use std::cell::{Cell, RefCell};

use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

#[allow(unused_imports)]
use libc;

// ─────────────────────────────────────────────────────────────────────────────
// WeightResidencySet -- MTLResidencySet lifecycle management
//
// Goal: tell macOS that weight buffers (GGUF mmaps) are needed by the GPU permanently
// and must not be evicted into compressed memory. This closes the 5.7x RSS gap
// between default allocation and llama.cpp for the same model.
//
// Pattern from llama.cpp ggml-metal-device.m:
// - A per-buffer residency set is created at the init of each Metal buffer
// - requestResidency() right after set creation -- memory is permanently wired
// - A background thread with a keep-alive interval (default 3 min) keeps the
//   residency alive via periodic requestResidency() every 500ms
//
// Our adaptation for Rust/candle (no per-buffer sets, one set per model):
// - WeightResidencySet is created at model init
// - new_buffer_no_copy adds each weight buffer via addAllocation()
// - commit_and_request() is called after all weights are loaded
// - A background Rust thread calls requestResidency() every HEARTBEAT_INTERVAL_S
//   seconds to prevent macOS eviction while idle
// - Drop calls endResidency() + removeAllAllocations() and joins the thread
//
// macOS version: MTLResidencySet is available since macOS 15.0. Checked at runtime
// via sysctl kern.osproductversion. If < 15 or env CANDLE_DISABLE_RESIDENCY_SET=1
// it returns None and behavior is unchanged.
// ─────────────────────────────────────────────────────────────────────────────

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLAllocation, MTLDevice as MTLDeviceProtocol, MTLResidencySet, MTLResidencySetDescriptor};
use objc2_foundation::NSString;

/// Heartbeat interval for residency requests (seconds).
/// llama.cpp uses 500ms; we take 30s -- enough to prevent eviction,
/// with less CPU overhead.
const HEARTBEAT_INTERVAL_S: u64 = 30;

/// Checks MTLResidencySet support at runtime (macOS 15+).
/// Result is cached in a OnceLock -- a single check on the first call.
fn supports_residency_set() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // Fast path via env override
        if std::env::var("CANDLE_DISABLE_RESIDENCY_SET").or_else(|_| std::env::var("YTTRI_DISABLE_RESIDENCY_SET"))
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            return false;
        }
        // macOS version check via sysctl kern.osproductversion
        // MTLResidencySet is available from 15.0; we are on 26.x -- always true,
        // but we check correctly for portability.
        macos_version_ge_15()
    })
}

/// Returns true if macOS >= 15.0 per kern.osproductversion.
fn macos_version_ge_15() -> bool {
    use std::ffi::CString;
    let mut buf = [0u8; 64];
    let mut size = buf.len();
    let key = match CString::new("kern.osproductversion") {
        Ok(k) => k,
        Err(_) => return true, // fallback: assume supported
    };
    let ret = unsafe {
        libc::sysctlbyname(
            key.as_ptr(),
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return true; // sysctlbyname failed — assume supported
    }
    // Parse "X.Y.Z" → major version
    let s = std::str::from_utf8(&buf[..size.saturating_sub(1)]).unwrap_or("26.0");
    let major: u32 = s.split('.').next().and_then(|m| m.parse().ok()).unwrap_or(26);
    major >= 15
}

/// Newtype wrapper making MTLResidencySet Send+Sync.
///
/// Safety: MTLResidencySet is thread-safe per Apple docs --
/// requestResidency/endResidency/addAllocation/commit can be called
/// from any thread. All ObjC retain/release operations in Retained<>
/// are atomic.
struct SendableRset(Retained<ProtocolObject<dyn MTLResidencySet>>);
unsafe impl Send for SendableRset {}
unsafe impl Sync for SendableRset {}

impl std::ops::Deref for SendableRset {
    type Target = ProtocolObject<dyn MTLResidencySet>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Internal residency set state -- holds the Objective-C object.
/// Wrapped in a Mutex so addAllocation/commit/requestResidency can be called
/// from different threads (although typically this happens sequentially).
struct ResidencyInner {
    rset: SendableRset,
    /// true = requestResidency() has been called, memory is wired.
    resident: bool,
}

// Safety: ResidencyInner contains only SendableRset (Send+Sync) and a bool.
unsafe impl Send for ResidencyInner {}
unsafe impl Sync for ResidencyInner {}

/// Public handle to the model's residency set.
///
/// Created via `MetalDevice::new_weight_residency_set()`.
/// Usage:
/// 1. Load weights via `new_buffer_no_copy` -- each buffer is automatically
///    added to the set if the device holds this WeightResidencySet.
/// 2. Call `commit_and_request()` after all weights are loaded.
/// 3. Keep the `Arc<WeightResidencySet>` alive while the model is in memory.
///    Drop() automatically calls endResidency().
pub struct WeightResidencySet {
    inner: Mutex<ResidencyInner>,
    /// Stop signal for the background heartbeat thread.
    stop: Arc<std::sync::atomic::AtomicBool>,
    /// JoinHandle for graceful shutdown on Drop.
    thread_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

unsafe impl Send for WeightResidencySet {}
unsafe impl Sync for WeightResidencySet {}

impl WeightResidencySet {
    /// Create a new WeightResidencySet on the given MTLDevice.
    /// Returns None if macOS < 15 or CANDLE_DISABLE_RESIDENCY_SET=1.
    pub fn new(device: &Device) -> Option<Arc<Self>> {
        if !supports_residency_set() {
            return None;
        }
        let desc = MTLResidencySetDescriptor::new();
        desc.setLabel(Some(&NSString::from_str("weights-residency")));
        let raw_device: &ProtocolObject<dyn MTLDeviceProtocol> =
            ProtocolObject::from_ref(device.as_ref());
        let rset = raw_device.newResidencySetWithDescriptor_error(&desc).ok()?;
        let rset = SendableRset(rset);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = stop.clone();
        // For the heartbeat thread we create a separate Arc<SendableRset>
        // so we do not clone the whole structure.
        let rset_for_thread = Arc::new(SendableRset(rset.0.clone()));
        // Background heartbeat thread -- analogous to llama.cpp background dispatch.
        // Periodically calls requestResidency() every HEARTBEAT_INTERVAL_S seconds.
        // macOS may "forget" the residency under memory pressure; the heartbeat
        // periodically reminds the OS that the buffers are needed by the GPU.
        // The heartbeat thread requests requestResidency() every 30s
        // only if CANDLE_WIRE_WEIGHTS=1. Otherwise the thread sleeps forever.
        let wire_weights = std::env::var("CANDLE_WIRE_WEIGHTS").or_else(|_| std::env::var("YTTRI_WIRE_WEIGHTS")).map(|v| v == "1").unwrap_or(false);
        let handle = std::thread::Builder::new()
            .name("residency-heartbeat".to_string())
            .spawn(move || {
                if !wire_weights {
                    // By default the heartbeat is inactive -- requestResidency is not called.
                    return;
                }
                while !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_secs(HEARTBEAT_INTERVAL_S));
                    if !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                        rset_for_thread.requestResidency();
                    }
                }
            })
            .ok();
        Some(Arc::new(Self {
            inner: Mutex::new(ResidencyInner { rset, resident: false }),
            stop,
            thread_handle: Mutex::new(handle),
        }))
    }

    /// Add a Metal buffer to the residency set (uncommitted).
    ///
    /// Called automatically from `MetalDevice::new_buffer_no_copy`.
    /// Threshold: only buffers >= 1 MiB (weight tensors, not small intermediates).
    pub fn add_buffer(&self, buffer: &Buffer) {
        if buffer.length() < 1024 * 1024 {
            return;
        }
        if let Ok(inner) = self.inner.lock() {
            let mtl_alloc: &ProtocolObject<dyn MTLAllocation> =
                ProtocolObject::from_ref(buffer.as_ref());
            inner.rset.addAllocation(mtl_alloc);
        }
    }

    /// Commit all added allocations (without requestResidency).
    ///
    /// Must be called ONCE after all model weights are loaded.
    /// commit() registers buffers in the residency set without forcibly wiring them.
    ///
    /// To force wiring, call `request_residency()` separately.
    /// Default: commit() only, without requestResidency() -- otherwise Physical footprint +500 MB.
    pub fn commit_and_request(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.rset.commit();
            // requestResidency() is intentionally NOT called here.
            // It wires memory into the GPU (non-evictable), increasing Physical footprint by ~500 MB.
            // Instead macOS manages eviction itself -- pages become resident
            // on demand at first GPU access.
            // For explicit wiring: CANDLE_WIRE_WEIGHTS=1 (see request_residency()).
            if std::env::var("CANDLE_WIRE_WEIGHTS").or_else(|_| std::env::var("YTTRI_WIRE_WEIGHTS")).map(|v| v == "1").unwrap_or(false) {
                inner.rset.requestResidency();
                inner.resident = true;
            }
            let alloc_size = inner.rset.allocatedSize();
            let alloc_mb = alloc_size / (1024 * 1024);
            eprintln!(
                "[residency] MTLResidencySet committed: {} allocations, ~{} MB tracked",
                inner.rset.allocationCount(),
                alloc_mb,
            );
        }
    }

    /// Return true if residency has already been requested.
    pub fn is_resident(&self) -> bool {
        self.inner.lock().map(|g| g.resident).unwrap_or(false)
    }

    /// Forcibly end residency (normally called via Drop).
    fn end_residency_inner(inner: &mut ResidencyInner) {
        if inner.resident {
            inner.rset.endResidency();
            inner.rset.removeAllAllocations();
            inner.rset.commit();
            inner.resident = false;
        }
    }
}

impl Drop for WeightResidencySet {
    fn drop(&mut self) {
        // Stop the background thread
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Ok(mut guard) = self.thread_handle.lock() {
            if let Some(handle) = guard.take() {
                // We do not block on join -- the thread will finish on its own
                // after the current sleep interval.
                let _ = handle.join();
            }
        }
        // End residency
        if let Ok(mut inner) = self.inner.lock() {
            Self::end_residency_inner(&mut inner);
        }
    }
}

use super::MetalError;

// ─────────────────────────────────────────────────────────────────────────────
// Allocation tracing API
//
// Default off -- no overhead in the hot path when tracing is inactive.
// Activated via begin_allocation_trace() only during a calibration
// forward. The result is collected via end_allocation_trace().
// ─────────────────────────────────────────────────────────────────────────────

thread_local! {
    static TRACE_ALLOC: Cell<bool> = const { Cell::new(false) };
    static TRACE_LOG: RefCell<Vec<TraceEntry>> = RefCell::new(Vec::new());
}

/// A single record of a Metal buffer allocation during a calibration forward.
#[derive(Clone, Debug)]
pub struct TraceEntry {
    /// Requested size (raw, before round-up).
    pub size: usize,
    /// Actual buffer size after buf_size() round-up.
    pub rounded_size: usize,
    /// Allocation moment (monotonic time).
    pub timestamp: std::time::Instant,
    /// true = buffer taken from the pool (reuse), false = new MTLBuffer.
    pub from_pool: bool,
}

/// Start allocation tracing. Clears the previous log.
/// Call before a calibration forward, after warmup.
pub fn begin_allocation_trace() {
    TRACE_LOG.with(|l| l.borrow_mut().clear());
    TRACE_ALLOC.with(|c| c.set(true));
}

/// Stop tracing and return the collected log.
/// Returns all TraceEntry since begin_allocation_trace().
pub fn end_allocation_trace() -> Vec<TraceEntry> {
    TRACE_ALLOC.with(|c| c.set(false));
    TRACE_LOG.with(|l| std::mem::take(&mut *l.borrow_mut()))
}

/// Check whether tracing is active on the current thread.
#[inline(always)]
pub fn allocation_trace_active() -> bool {
    TRACE_ALLOC.with(|c| c.get())
}

#[inline(always)]
fn record_trace(size: usize, rounded: usize, from_pool: bool) {
    TRACE_LOG.with(|l| {
        l.borrow_mut().push(TraceEntry {
            size,
            rounded_size: rounded,
            timestamp: std::time::Instant::now(),
            from_pool,
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Scratch arena thread_local activation
//
// ACTIVE_ARENA holds the current scratch arena for this thread (if active).
// It is activated via MetalDevice::activate_scratch_arena() before the prefill loop,
// and deactivated via deactivate_scratch_arena() after the final fence.
//
// Default: None -- the arena is inactive, allocate_buffer uses the regular pool.
// ─────────────────────────────────────────────────────────────────────────────

thread_local! {
    static ACTIVE_ARENA: RefCell<Option<Arc<ScratchArena>>> = const { RefCell::new(None) };
    /// Active UnifiedScratchArena for Phase 3c offset dispatch.
    /// None = disabled (default). Activated via activate_unified_arena().
    static ACTIVE_UNIFIED_ARENA: RefCell<Option<Arc<UnifiedScratchArena>>> = const { RefCell::new(None) };
    /// If true -- the next allocate_buffer skips the arena fast path.
    /// Used to exclude long-lived allocations (KV cache) from the arena.
    /// Automatically reset on every read (one-shot flag).
    static SKIP_ARENA_NEXT: Cell<bool> = const { Cell::new(false) };
}

/// Mark the next allocation as "not for the arena" (a long-lived buffer).
///
/// Call before creating KV cache buffers (`Tensor::zeros`) or other
/// long-lived buffers that should not occupy arena slots.
///
/// The flag is automatically reset after a single `allocate_buffer` call.
pub fn skip_arena_next_alloc() {
    SKIP_ARENA_NEXT.with(|s| s.set(true));
}

/// Unique identifier for metal devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    pub(crate) id: DeviceId,

    /// Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    pub(crate) device: Device,

    pub(crate) commands: Arc<Commands>,

    /// Simple allocator struct.
    /// The buffers are stored in size buckets since ML tends to use similar shapes over and over.
    /// We store the buffers in [`Arc`] because it's much faster than Obj-c internal ref counting
    /// (could be linked to FFI communication overhead).
    ///
    /// Whenever a buffer has a strong_count==1, we can reuse it, it means it was dropped in the
    /// graph calculation, and only we the allocator kept a reference to it, therefore it's free
    /// to be reused. However, in order for this to work, we need to guarantee the order of
    /// operation, so that this buffer is not being used by another kernel at the same time.
    /// Arc is the CPU reference count, it doesn't mean anything on the GPU side of things.
    ///
    /// Whenever we actually allocate a new buffer, we make a full sweep to clean up unused buffers
    /// (strong_count = 1).
    pub(crate) buffers: Arc<RwLock<BufferMap>>,

    /// Same as `buffers` but uses `PRIVATE_RESOURCE_OPTIONS` (StorageModePrivate on macOS).
    /// Intermediate compute buffers don't need CPU access so Private avoids coherency overhead.
    pub(crate) private_buffers: Arc<RwLock<BufferMap>>,

    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`candle_metal_kernels`]
    pub(crate) kernels: Arc<Kernels>,
    /// Seed for random number generation.
    pub(crate) seed: Arc<Mutex<Buffer>>,
    /// Last seed value set on this device.
    pub(crate) seed_value: Arc<RwLock<u64>>,
    pub(crate) completion_aware_pool: bool,

    /// MTLResidencySet for weight buffers.
    ///
    /// Created via `new_weight_residency_set()` after device init.
    /// None = macOS < 15 or CANDLE_DISABLE_RESIDENCY_SET=1.
    /// `new_buffer_no_copy` automatically adds buffers to this set.
    ///
    /// Arc<Mutex<>> so it can be mutated via &self (MetalDevice is Clone).
    pub(crate) weight_residency: Arc<Mutex<Option<Arc<WeightResidencySet>>>>,
    /// Residency set registered on the command queue.
    pub(crate) residency_set: Arc<ResidencySet>,
}

// Resource options used for creating buffers. Shared storage mode allows both CPU and GPU to access the buffer.
pub const RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);
// Resource options used for `new_private_buffer`. This uses `private` where supported.
#[cfg(target_os = "ios")]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = RESOURCE_OPTIONS;
#[cfg(not(target_os = "ios"))]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = objc2_metal::MTLResourceOptions(
    MTLResourceOptions::StorageModePrivate.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.id)
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    #[cfg(all(feature = "ug", not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<ComputePipeline> {
        let mut buf = vec![];
        candle_ug::metal::code_gen::gen(&mut buf, func_name, &kernel)?;
        let metal_code = String::from_utf8(buf)?;
        let lib = self
            .device
            .new_library_with_source(&metal_code, None)
            .map_err(MetalError::from)?;
        let func = lib
            .get_function(func_name, None)
            .map_err(MetalError::from)?;
        let pl = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(MetalError::from)?;
        Ok(pl)
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn metal_device(&self) -> &Device {
        &self.device
    }

    fn drop_unused_buffers(&self) -> Result<()> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        for subbuffers in buffers.values_mut() {
            subbuffers.retain(|s| {
                if Arc::strong_count(s) == 1 {
                    self.residency_set.remove(s);
                    false
                } else {
                    true
                }
            });
        }
        let mut private_buffers = self.private_buffers.write().map_err(MetalError::from)?;
        for subbuffers in private_buffers.values_mut() {
            subbuffers.retain(|s| {
                if Arc::strong_count(s) == 1 {
                    self.residency_set.remove(s);
                    false
                } else {
                    true
                }
            });
        }
        Ok(())
    }

    /// Flush all unused buffers from the Metal buffer pool.
    ///
    /// Removes buffers with `strong_count == 1` (only the pool holds a reference)
    /// and removes empty size buckets. Call after model loading to reclaim memory
    /// used by intermediate buffers (dequantization temporaries, etc.).
    pub fn flush_buffers(&self) -> Result<()> {
        self.drop_unused_buffers()?;
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        buffers.retain(|_, v| !v.is_empty());
        Ok(())
    }

    /// Aggressive buffer pool cleanup.
    ///
    /// Unlike `flush_buffers()` this fully clears all size buckets
    /// after waiting for in-flight GPU commands to finish. Also clears
    /// the compiled Metal library and pipeline state object caches.
    ///
    /// # Important notes
    ///
    /// - **Cold-start spike**: after the call all Metal shaders will be recompiled
    ///   on next use (~50-200ms per source). Call only at engine teardown,
    ///   not between inferences.
    ///
    /// - **Live Arc references**: buffers that have `Arc` references from
    ///   live tensors (`MetalStorage`) will not be physically released -- only
    ///   removed from the pool. New allocations will create new buffers instead
    ///   of reusing. Call after dropping all model tensors.
    pub fn purge_buffer_pool(&self) -> Result<()> {
        self.wait_until_completed()?;
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        buffers.clear();
        let _ = self.kernels.clear_caches();
        Ok(())
    }

    /// Returns buffer pool statistics: `(total_buffers, total_bytes, unused_buffers, unused_bytes)`.
    ///
    /// Unused buffers have `strong_count == 1` and can be reclaimed by [`flush_buffers`].
    pub fn buffer_pool_stats(&self) -> Result<(usize, usize, usize, usize)> {
        let buffers = self.buffers.read().map_err(MetalError::from)?;
        let mut total_count = 0usize;
        let mut total_bytes = 0usize;
        let mut unused_count = 0usize;
        let mut unused_bytes = 0usize;
        for subbuffers in buffers.values() {
            for buf in subbuffers {
                let len = buf.length();
                total_count += 1;
                total_bytes += len;
                if Arc::strong_count(buf) == 1 {
                    unused_count += 1;
                    unused_bytes += len;
                }
            }
        }
        Ok((total_count, total_bytes, unused_count, unused_bytes))
    }

    /// Command pool statistics:
    /// `(entries, in_flight_total, encoding_entries, total_compute_count)`.
    pub fn command_pool_stats(&self) -> Result<(usize, usize, usize, usize)> {
        Ok(self.commands.pool_stats().map_err(MetalError::from)?)
    }

    /// Kernel cache sizes: `(libraries, pipelines)`.
    pub fn kernel_cache_stats(&self) -> Result<(usize, usize)> {
        Ok(self.kernels.cache_stats().map_err(MetalError::from)?)
    }

    /// Summary of Metal runtime memory:
    /// `(current_allocated_bytes, recommended_max_working_set_bytes)`.
    pub fn metal_memory_stats(&self) -> (usize, usize) {
        (
            self.device.current_allocated_size(),
            self.device.recommended_max_working_set_size(),
        )
    }

    pub fn completion_aware_pool_enabled(&self) -> bool {
        self.completion_aware_pool
    }

    pub fn command_encoder<'a>(&'a self) -> Result<CommandsGuard<'a>> {
        let command_encoder = self.commands.command_encoder().map_err(MetalError::from)?;

        Ok(command_encoder)
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>> {
        let command_encoder = self
            .commands
            .blit_command_encoder()
            .map_err(MetalError::from)?;
        Ok(command_encoder)
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        self.commands
            .wait_until_completed()
            .map_err(MetalError::from)?;

        self.drop_unused_buffers()?;
        Ok(())
    }

    /// Commit and wait on the buffer holding the caller's work; safe for concurrent CPU readbacks.
    pub fn flush_and_wait_current(&self) -> Result<()> {
        self.commands
            .flush_and_wait_current()
            .map_err(MetalError::from)?;

        self.drop_unused_buffers()?;
        Ok(())
    }

    /// Fast sync optimized for single-threaded inference workloads.
    /// Only processes pool entries that have pending work, skipping empty ones.
    pub fn wait_until_completed_fast(&self) -> Result<()> {
        self.commands.flush_and_wait_fast().map_err(MetalError::from)?;
        Ok(())
    }

    /// Commit any pending command buffers to the GPU **without** waiting for
    /// completion. Enables CPU↔GPU pipelining when used inside hot loops
    /// (e.g. periodic flush at every N layers in prefill loop): CPU
    /// continues encoding the next batch of ops while GPU executes the
    /// committed work in parallel.
    ///
    /// **Hazard warning.** `flush_no_wait` does NOT clean up the buffer pool
    /// or wait for in-flight work. The buffer pool may hand out a buffer
    /// (matched by size and `Arc::strong_count == 1`) that GPU is still
    /// writing to in a previous CB — leading to data races. Use ONLY inside
    /// a `with_shared_encoder` scope or when the caller can guarantee no
    /// pool reuse touches buffers from in-flight CBs.
    ///
    /// The caller must still call `wait_until_completed()` (or
    /// `wait_until_completed_fast()`) before reading GPU outputs from CPU.
    pub fn flush_no_wait(&self) -> Result<()> {
        self.commands.flush().map_err(MetalError::from)?;
        Ok(())
    }

    /// Run a closure inside a graph-compute scope where all Metal compute
    /// ops accumulate into a single command buffer (one `commit` at scope
    /// exit) instead of the default behaviour of auto-flushing every
    /// `compute_per_buffer` ops.
    ///
    /// Modeled on llama.cpp's `ggml_backend_metal_graph_compute`: a long
    /// sequence of ops (e.g. a model forward pass) is encoded into one
    /// CB. CPU encoding still runs serially, but the CPU↔GPU sync overhead
    /// from per-batch commits is removed. Buffer pool reuse stays safe
    /// because all ops live in the same CB and Metal's shared-storage
    /// hazards are enforced by hardware between encoders within a CB.
    ///
    /// Memory profile improves too: intermediate buffers go back to the
    /// pool as soon as their last `Tensor` is dropped (which happens
    /// continuously during encoding), so peak in-flight memory tracks
    /// the natural data-dependency depth of the graph rather than the
    /// raw layer count.
    ///
    /// Caveats:
    /// - The closure must complete its CPU side cleanly before any GPU
    ///   output is read; this function calls `wait_until_completed_fast`
    ///   on exit so the caller may read tensors right after.
    /// - Large CBs slow down the GPU's first-byte latency, so for very
    ///   small graphs (single-token decode) the default 50-op auto-flush
    ///   path is usually faster — keep `with_shared_encoder` for prefill.
    /// - Nesting is supported via thread_local stacking of the scope
    ///   limit (closure-local restore at exit).
    pub fn with_shared_encoder<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        // Effectively-infinite limit so finalize_entry never auto-flushes.
        // Real-world forwards are 200-300 ops; a million gives a huge margin.
        const SCOPE_LIMIT: usize = 1_000_000;

        let prev_limit = candle_metal_kernels::metal::commands::graph_scope_limit();
        candle_metal_kernels::metal::commands::set_graph_scope_limit(SCOPE_LIMIT);

        struct ScopeGuard {
            prev: usize,
        }
        impl Drop for ScopeGuard {
            fn drop(&mut self) {
                candle_metal_kernels::metal::commands::set_graph_scope_limit(self.prev);
            }
        }
        let _guard = ScopeGuard { prev: prev_limit };

        let result = f();

        // Final sync: CB commit + wait. After this pool reuse is safe,
        // GPU outputs are ready for CPU reads.
        self.wait_until_completed_fast()?;

        result
    }

    /// Like `with_shared_encoder` but does NOT wait for GPU completion on exit.
    ///
    /// All Metal ops inside the closure go into a single CB (no auto-commits).
    /// The CB is committed when the pool entry is next flushed, but the caller
    /// MUST call `wait_until_completed_fast()` before reading any GPU outputs
    /// from CPU, or before any operation that might reuse buffers from this scope.
    ///
    /// Designed for multi-CB prefill: the prefill loop encodes all 32
    /// layers into one CB, then the caller waits once at the very end. This
    /// removes ~6 unnecessary commit→swap cycles during the prefill encoding
    /// phase, reducing CPU dispatch overhead.
    pub fn with_single_cb_scope<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        const SCOPE_LIMIT: usize = 1_000_000;

        let prev_limit = candle_metal_kernels::metal::commands::graph_scope_limit();
        candle_metal_kernels::metal::commands::set_graph_scope_limit(SCOPE_LIMIT);

        struct ScopeGuard {
            prev: usize,
        }
        impl Drop for ScopeGuard {
            fn drop(&mut self) {
                candle_metal_kernels::metal::commands::set_graph_scope_limit(self.prev);
            }
        }
        let _guard = ScopeGuard { prev: prev_limit };

        // No wait! Caller handles GPU sync later.
        f()
    }

    /// Get and reset accumulated sync timing stats from flush_and_wait_fast.
    /// Returns: (count, sem_us, lock_us, commit_us, wait_us, total_us)
    pub fn take_sync_timings(&self) -> (u64, u64, u64, u64, u64, u64) {
        Commands::take_sync_timings()
    }

    /// Static version — no device instance needed (timings are thread-local).
    pub fn take_sync_timings_static() -> (u64, u64, u64, u64, u64, u64) {
        Commands::take_sync_timings()
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    // ─────────────────────────────────────────────────────────────────────
    // WeightResidencySet API
    // ─────────────────────────────────────────────────────────────────────

    /// Create a WeightResidencySet for this device and set it as active.
    /// After this every `new_buffer_no_copy` (GGUF weights) automatically
    /// adds the buffer to the set.
    ///
    /// Returns an `Arc<WeightResidencySet>` -- keep it alive while the model lives.
    /// Call `commit_weight_residency()` after all weights are loaded.
    ///
    /// If macOS < 15 or CANDLE_DISABLE_RESIDENCY_SET=1 -- returns None.
    pub fn new_weight_residency_set(&self) -> Option<Arc<WeightResidencySet>> {
        let rset = WeightResidencySet::new(&self.device)?;
        if let Ok(mut guard) = self.weight_residency.lock() {
            *guard = Some(rset.clone());
        }
        Some(rset)
    }

    /// Commit the residency set and request wiring for all added
    /// weight buffers. Call ONCE after all model weights are loaded.
    ///
    /// No-op if no residency set is installed.
    pub fn commit_weight_residency(&self) {
        if let Ok(guard) = self.weight_residency.lock() {
            if let Some(ref rset) = *guard {
                rset.commit_and_request();
            }
        }
    }

    /// Get a clone of the active WeightResidencySet (if any).
    pub fn weight_residency_set(&self) -> Option<Arc<WeightResidencySet>> {
        self.weight_residency.lock().ok()?.as_ref().cloned()
    }

    /// Registers buffers in the device's residency set, keeping them
    /// permanently GPU-resident instead of paying per-command-buffer residency
    /// bookkeeping. Useful for buffers candle did not allocate, e.g.
    /// `newBufferWithBytesNoCopy` views over an mmap'd weights file. No-op on
    /// systems without residency-set support.
    pub fn register_buffers<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) {
        self.residency_set.insert_batch(bufs);
    }

    /// Unregisters buffers previously passed to `register_buffers`, releasing
    /// the set's retain so they can be deallocated. Only unregister buffers
    /// you registered yourself, after GPU work referencing them has completed.
    pub fn unregister_buffers<'a>(&self, bufs: impl IntoIterator<Item = &'a Buffer>) {
        self.residency_set.remove_batch(bufs);
    }

    /// Returns a builder for buffer allocation. See `BufferBuilder`.
    pub fn new_buffer_builder(&self) -> BufferBuilder<'_> {
        BufferBuilder::new(self)
    }

    /// Creates a new buffer (not necessarily zeroed).
    ///
    /// Uses StorageModePrivate on macOS for faster GPU access (no CPU coherency overhead).
    /// Falls back to StorageModeShared on iOS where Private is not always available.
    pub fn new_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = element_count * dtype.size_in_bytes();
        let mut buffers = self.private_buffers.write().map_err(MetalError::from)?;
        if let Some(b) = find_available_buffer(size, &buffers, None) {
            return Ok(b.clone());
        }
        let size = buf_size(size);
        let subbuffers = buffers.entry(size).or_insert(vec![]);

        let new_buffer = self
            .device
            .new_buffer(size, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let new_buffer = Arc::new(new_buffer);
        self.residency_set.insert(&new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    /// Creates a new private buffer (not necessarily zeroed).
    ///
    /// This is intentionally not in the Metal buffer pool to allow the efficient implementation of persistent buffers.
    pub fn new_private_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = element_count * dtype.size_in_bytes();
        let buffer = self
            .device
            .new_buffer(size, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let buffer = Arc::new(buffer);
        self.residency_set.insert(&buffer);
        Ok(buffer)
    }

    /// Creates a new buffer from data.
    ///
    /// Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
    /// allocates the buffer and copies over the existing data before returning the MTLBuffer.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data);
        let new_buffer = self
            .device
            .new_buffer_with_data(data.as_ptr().cast(), size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;

        let subbuffers = buffers.entry(size).or_insert(vec![]);

        let new_buffer = Arc::new(new_buffer);
        self.residency_set.insert(&new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    /// Creates a Metal buffer without copying data (zero-copy) from mmap'd memory.
    ///
    /// The buffer is NOT added to the MetalDevice buffer pool -- it is tied to the mmap
    /// and must not be reused by other operations.
    ///
    /// If a `weight_residency` set is installed,
    /// the buffer is automatically added via `addAllocation()` (uncommitted).
    /// Call `commit_weight_residency()` after all weights are loaded.
    ///
    /// Requirements:
    /// - `ptr` MUST be page-aligned (mmap guarantees this)
    /// - `len` MUST be a multiple of the page size (otherwise Metal returns an error)
    /// - The caller MUST guarantee that the mmap outlives the buffer
    pub fn new_buffer_no_copy(
        &self,
        ptr: *mut std::ffi::c_void,
        len: usize,
    ) -> Result<Arc<Buffer>> {
        let new_buffer = self
            .device
            .new_buffer_no_copy(ptr, len, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let new_buffer = Arc::new(new_buffer);
        // Add to the residency set if one is installed.
        // Only large buffers (>= 1 MiB) -- weight tensors, not intermediates.
        if let Ok(guard) = self.weight_residency.lock() {
            if let Some(ref rset) = *guard {
                rset.add_buffer(&new_buffer);
            }
        }
        Ok(new_buffer)
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<Buffer>> {
        let buffer = self.allocate_buffer(size_in_bytes)?;
        let mut blit = self.blit_command_encoder()?;
        blit.set_label("zeros");
        blit.fill_buffer(&buffer, (0, buffer.length()), 0);
        /*
        // Alternative impl
        if size_in_bytes > 0 {
            let encoder = self.command_encoder()?;
            call_const_fill(
                &self.device,
                &encoder,
                &self.kernels,
                "fill_u8",
                size_in_bytes,
                &buffer,
                0u8,
            )
            .map_err(crate::Error::wrap)?;
        }
        */
        Ok(buffer)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Scratch arena API
    // ─────────────────────────────────────────────────────────────────────

    /// Create a scratch arena with the given slot sizes.
    ///
    /// `slot_sizes` -- byte sizes of each slot (usually power-of-two,
    /// from Phase 1 trace results).
    ///
    /// The arena allocates all MTLBuffers up front at creation.
    /// Returns an `Arc<ScratchArena>` ready to be activated via
    /// `activate_scratch_arena`.
    pub fn create_scratch_arena(&self, slot_sizes: &[usize]) -> Result<Arc<ScratchArena>> {
        Ok(Arc::new(
            ScratchArena::new(&self.device, slot_sizes).map_err(MetalError::from)?,
        ))
    }

    // ─────────────────────────────────────────────────────────────────────
    // UnifiedScratchArena API
    // ─────────────────────────────────────────────────────────────────────

    /// Create a UnifiedScratchArena (one big MTLBuffer + bump allocator).
    ///
    /// `capacity` -- total size in bytes (recommended ~764 MB for Qwen3.5-2B).
    /// Allocates a single MTLBuffer at creation.
    pub fn create_unified_arena(
        &self,
        capacity: usize,
    ) -> Result<Arc<UnifiedScratchArena>> {
        Ok(Arc::new(
            UnifiedScratchArena::new(&self.device, capacity).map_err(MetalError::from)?,
        ))
    }

    /// Activate the unified arena for the current thread.
    ///
    /// After activation `try_acquire_unified` can be used for offset dispatch.
    /// Default off -- no effect on `allocate_buffer` until then.
    pub fn activate_unified_arena(&self, arena: Arc<UnifiedScratchArena>) {
        ACTIVE_UNIFIED_ARENA.with(|a| *a.borrow_mut() = Some(arena));
    }

    /// Deactivate the unified arena for the current thread.
    pub fn deactivate_unified_arena(&self) {
        ACTIVE_UNIFIED_ARENA.with(|a| *a.borrow_mut() = None);
    }

    /// Reset the unified arena bump offset (after a GPU fence).
    ///
    /// Calling `wait_until_completed_fast()` BEFORE reset is MANDATORY.
    pub fn reset_unified_arena(&self) {
        ACTIVE_UNIFIED_ARENA.with(|a| {
            if let Some(arena) = a.borrow().as_ref() {
                arena.reset();
            }
        });
    }

    /// Try to allocate from the unified arena.
    ///
    /// Returns `(Arc<Buffer>, offset_in_bytes)` on success.
    /// Returns `None` if the arena is inactive or exhausted.
    ///
    /// # Usage
    ///
    /// ```ignore
    /// if let Some(alloc) = device.try_acquire_unified(size) {
    ///     call_kernel(..., &alloc.buffer, alloc.offset_in_bytes);
    ///     // buffer and offset are passed directly to the encoder without creating a separate MTLBuffer
    /// } else {
    ///     let buffer = device.allocate_buffer(size)?;
    ///     call_kernel(..., &buffer, 0);
    /// }
    /// ```
    pub fn try_acquire_unified(
        &self,
        size: usize,
    ) -> Option<candle_metal_kernels::metal::UnifiedAlloc> {
        ACTIVE_UNIFIED_ARENA.with(|a| {
            a.borrow()
                .as_ref()
                .and_then(|arena| arena.try_acquire(size))
        })
    }

    /// Allocate from the unified arena, returning `(Arc<Buffer>, offset)`.
    ///
    /// If the unified arena is active -- bump-allocate from it.
    /// If the arena is inactive or exhausted -- fall back to `allocate_buffer(size)` with offset=0.
    ///
    /// The result is used to create `MetalStorage::new_with_offset`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee a GPU fence (`wait_until_completed_fast`) before
    /// the next `reset_unified_arena()`. Otherwise the GPU may write to an offset
    /// concurrently with a new kernel at the same offset.
    pub fn new_buffer_unified(
        &self,
        element_count: usize,
        dtype: crate::DType,
        _name: &str,
    ) -> Result<(Arc<Buffer>, usize)> {
        let size = element_count * dtype.size_in_bytes();
        // Try the unified arena first.
        if let Some(alloc) = self.try_acquire_unified(size) {
            return Ok((alloc.buffer, alloc.offset_in_bytes));
        }
        // Fallback: regular pool, offset = 0.
        let buf = self.allocate_buffer(size)?;
        Ok((buf, 0))
    }

    /// Activate the scratch arena for the current thread.
    ///
    /// After the call `allocate_buffer` first tries to acquire a slot
    /// from the arena (lock-free via Arc::strong_count). If the arena has no
    /// suitable free slot -- it falls back to the regular pool.
    ///
    /// Call before the prefill loop. The paired deactivation --
    /// `deactivate_scratch_arena()` -- is MANDATORY after the final GPU fence.
    pub fn activate_scratch_arena(&self, arena: Arc<ScratchArena>) {
        ACTIVE_ARENA.with(|a| *a.borrow_mut() = Some(arena));
    }

    /// Deactivate the scratch arena for the current thread.
    ///
    /// After the call `allocate_buffer` uses only the regular pool.
    /// Always call after the prefill loop (even on panic -- use an RAII guard).
    pub fn deactivate_scratch_arena(&self) {
        ACTIVE_ARENA.with(|a| *a.borrow_mut() = None);
    }

    /// Get the currently active scratch arena (for diagnostics).
    pub fn active_scratch_arena(&self) -> Option<Arc<ScratchArena>> {
        ACTIVE_ARENA.with(|a| a.borrow().clone())
    }

    /// The critical allocator algorithm
    pub fn allocate_buffer(&self, size: usize) -> Result<Arc<Buffer>> {
        // ─── SCRATCH ARENA FAST PATH ───────────────────
        // Check the arena BEFORE the pool lock. Lock-free: only an atomic
        // strong_count check inside try_acquire.
        // Default off: ACTIVE_ARENA holds None -> a free borrow().
        //
        // SKIP_ARENA_NEXT: if set -- skip the arena for this allocation.
        // Used to exclude long-lived buffers (KV cache) from the arena.
        let skip_arena = SKIP_ARENA_NEXT.with(|s| {
            let v = s.get();
            if v {
                s.set(false); // one-shot flag
            }
            v
        });
        if !skip_arena {
            if let Some(buf) = ACTIVE_ARENA.with(|a| {
                a.borrow()
                    .as_ref()
                    .and_then(|arena| arena.try_acquire(size))
            }) {
                // Arena hit: record in the trace as from_pool=false
                // so Phase 1/2 trace can distinguish arena vs pool vs new.
                if allocation_trace_active() {
                    record_trace(size, buf.length(), false);
                }
                return Ok(buf);
            }
        }
        // ─── EXISTING POOL PATH (unchanged) ─────────────────────────────

        // Check tracing once -- no branch misprediction in the hot path when off.
        let trace_on = allocation_trace_active();

        let completed_command_buffer_id = if self.completion_aware_pool {
            let commands = &self.commands;
            Some(commands.completed_command_buffer_id())
        } else {
            None
        };
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        if let Some(b) = find_available_buffer(size, &buffers, completed_command_buffer_id) {
            // Cloning also ensures we increment the strong count
            if trace_on {
                record_trace(size, b.length(), true);
            }
            return Ok(b.clone());
        }
        let rounded = buf_size(size);
        let subbuffers = buffers.entry(rounded).or_insert(vec![]);

        let new_buffer = self
            .device
            .new_buffer(rounded, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let new_buffer = Arc::new(new_buffer);
        self.residency_set.insert(&new_buffer);
        subbuffers.push(new_buffer.clone());
        if trace_on {
            record_trace(size, rounded, false);
        }
        Ok(new_buffer)
    }

    /// Create a metal GPU capture trace on [`path`].
    pub fn capture<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
        let descriptor = MTLCaptureDescriptor::new();
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        descriptor.set_capture_device(self.device().as_ref());
        if path.as_ref().is_absolute() {
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        } else {
            let path = std::env::current_dir()?.join(path);
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        }
        capture
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|e| MetalError::from(e.to_string()))?;
        Ok(())
    }

    /// Stop the current GPU capture.
    pub fn stop_capture(&self) {
        unsafe { MTLCaptureManager::sharedCaptureManager() }.stopCapture();
    }
}

fn buf_size(size: usize) -> usize {
    size.next_power_of_two()
}

/// Applies the [`BufferBuilder`] label, clearing any stale label on a reused pooled buffer.
#[cfg(feature = "metal-debug-labels")]
#[inline]
fn buffer_label(buffer: &Buffer, label: Option<&str>) {
    buffer.set_label(label.unwrap_or("unlabeled"));
}
#[cfg(not(feature = "metal-debug-labels"))]
#[inline]
fn buffer_label(_buffer: &Buffer, _label: Option<&str>) {}

type DataUpload<'a> = Box<dyn FnOnce(&MetalDevice) -> Result<Arc<Buffer>> + 'a>;

enum BufferInit<'a> {
    Typed { elem_count: usize, dtype: DType },
    Size(usize),
    Zeros(usize),
    Data(DataUpload<'a>),
}

/// Builder for `MTLBuffer` allocations; pool reuse handled by [`MetalDevice`].
pub struct BufferBuilder<'a> {
    device: &'a MetalDevice,
    label: Option<&'a str>,
}

/// [`BufferBuilder`] with an init kind set; `build()` lives here.
pub struct ReadyBufferBuilder<'a> {
    device: &'a MetalDevice,
    init: BufferInit<'a>,
    label: Option<&'a str>,
}

impl<'a> BufferBuilder<'a> {
    fn new(device: &'a MetalDevice) -> Self {
        Self {
            device,
            label: None,
        }
    }

    /// Allocate elem_count * dtype size bytes, uninitialized, private storage.
    pub fn with_size_for(self, elem_count: usize, dtype: DType) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Typed { elem_count, dtype })
    }

    /// Allocate size bytes, uninitialized, shared storage.
    pub fn with_size(self, size: usize) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Size(size))
    }

    /// Allocate size bytes, zero-filled, shared storage. Pool rounding may make
    /// the allocation larger than size; the extra bytes are also zeroed.
    pub fn with_zeros(self, size: usize) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Zeros(size))
    }

    /// Allocate a shared buffer initialized from data. Always allocates; does not
    /// reuse the pool.
    pub fn with_data<T>(self, data: &'a [T]) -> ReadyBufferBuilder<'a> {
        self.ready(BufferInit::Data(Box::new(move |device| {
            device.new_buffer_with_data(data)
        })))
    }

    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    #[inline]
    fn ready(self, init: BufferInit<'a>) -> ReadyBufferBuilder<'a> {
        ReadyBufferBuilder {
            device: self.device,
            init,
            label: self.label,
        }
    }
}

impl<'a> ReadyBufferBuilder<'a> {
    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    pub fn build(self) -> Result<Arc<Buffer>> {
        let buffer = match self.init {
            BufferInit::Typed { elem_count, dtype } => {
                self.device.new_buffer(elem_count, dtype, "")?
            }
            BufferInit::Size(size) => self.device.allocate_buffer(size)?,
            BufferInit::Zeros(size) => self.device.allocate_zeros(size)?,
            BufferInit::Data(upload) => upload(self.device)?,
        };
        buffer_label(&buffer, self.label);
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buf_size_exact_powers_of_two() {
        assert_eq!(buf_size(1), 1);
        assert_eq!(buf_size(2), 2);
        assert_eq!(buf_size(4), 4);
        assert_eq!(buf_size(8), 8);
        assert_eq!(buf_size(16), 16);
        assert_eq!(buf_size(1024), 1024);
    }

    #[test]
    fn test_buf_size_rounds_up() {
        assert_eq!(buf_size(3), 4);
        assert_eq!(buf_size(5), 8);
        assert_eq!(buf_size(6), 8);
        assert_eq!(buf_size(7), 8);
        assert_eq!(buf_size(9), 16);
        assert_eq!(buf_size(1000), 1024);
        assert_eq!(buf_size(1025), 2048);
    }

    #[test]
    fn test_buf_size_bf16_f16_scalar() {
        // BF16 and F16 are 2 bytes per element. A scalar tensor requests
        // a 2-byte buffer. This must not be rounded down to 1.
        assert_eq!(buf_size(2), 2);
    }
}

fn find_available_buffer(
    size: usize,
    buffers: &BufferMap,
    completed_command_buffer_id: Option<u64>,
) -> Option<Arc<Buffer>> {
    let mut best_buffer: Option<&Arc<Buffer>> = None;
    let mut best_buffer_size = usize::MAX;
    for (buffer_size, subbuffers) in buffers.iter() {
        if buffer_size >= &size && buffer_size < &best_buffer_size {
            for sub in subbuffers {
                let gpu_completed = completed_command_buffer_id
                    .map(|completed| sub.last_used_command_buffer_id() <= completed)
                    .unwrap_or(true);
                if Arc::strong_count(sub) == 1 && gpu_completed {
                    best_buffer = Some(sub);
                    best_buffer_size = *buffer_size;
                }
            }
        }
    }
    best_buffer.cloned()
}

pub(crate) fn completion_aware_pool_enabled_from_env() -> bool {
    std::env::var("CANDLE_METAL_COMPLETION_AWARE_POOL")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}
