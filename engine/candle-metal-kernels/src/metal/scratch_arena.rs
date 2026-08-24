// Pre-allocated scratch arena for intermediate Metal buffers.
//
// Slot lifecycle:
// 1. arena.try_acquire(size) -> Some(Arc<Buffer>) if a free slot is found
//    with capacity >= rounded(size). The slot strong_count becomes 2 (arena + caller).
// 2. Caller owns the Arc, passes it into Tensor / MetalStorage.
// 3. When the Tensor is dropped, Arc::drop -> strong_count goes back to 1.
// 4. If the caller invoked a GPU sync (wait_until_completed_fast), the GPU is done writing.
// 5. The next try_acquire sees strong_count == 1 -> reuse is safe.
//
// Safety: arena reuse is possible ONLY after an external fence.
// The caller must call wait_until_completed_fast() BEFORE
// the next allocate_buffer() so the GPU has finished writing to previous slots.
// This is guaranteed by the sync-per-4-layers point in the prefill loop.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::MetalKernelError;

use super::buffer::{Buffer, MTLResourceOptions};
use super::device::Device;

/// A single slot in the scratch arena.
///
/// `buffer` -- a pre-allocated MTLBuffer with StorageModeShared.
/// The slot is "free" when `Arc::strong_count(&buffer) == 1`
/// (only the arena holds a reference).
pub struct ArenaSlot {
    pub buffer: Arc<Buffer>,
    pub capacity: usize,
}

/// Pre-allocated scratch arena for intermediate Metal buffers.
///
/// All slots are allocated at creation via `new()`. `try_acquire()`
/// finds a free slot with sufficient capacity -- lock-free via
/// `Arc::strong_count`. On failure the caller falls back to the pool.
///
/// # Safety
///
/// `Send + Sync` are declared unsafe because the inner `Arc<Buffer>` is not Send
/// due to objc2 internals. Inference is single-threaded, so this is safe --
/// analogous to the existing `Commands` and `Buffer` in candle-metal-kernels.
///
/// GPU safety is guaranteed by an external fence: before reuse the caller must
/// call `MetalDevice::wait_until_completed_fast()`.
pub struct ScratchArena {
    /// Pre-allocated slots. The index is fixed for the arena's whole lifetime.
    pub slots: Vec<ArenaSlot>,
    /// Round-robin start for searching a free slot (optimization).
    /// Cell<usize> -- no mutex, inference is single-threaded.
    acquire_start: Cell<usize>,
}

unsafe impl Send for ScratchArena {}
unsafe impl Sync for ScratchArena {}

impl ScratchArena {
    /// Create an arena with the given slot sizes.
    ///
    /// Each element of `slot_sizes` becomes one MTLBuffer (StorageModeShared).
    /// Allocations happen immediately, at arena creation.
    ///
    /// `slot_sizes` must contain **exact byte sizes** (not rounded),
    /// usually power-of-two values from the Phase 1 trace.
    pub fn new(device: &Device, slot_sizes: &[usize]) -> Result<Self, MetalKernelError> {
        let opts = MTLResourceOptions::StorageModeShared;
        let mut slots = Vec::with_capacity(slot_sizes.len());
        for &sz in slot_sizes {
            let buf = device.new_buffer(sz, opts)?;
            slots.push(ArenaSlot {
                buffer: Arc::new(buf),
                capacity: sz,
            });
        }
        Ok(Self {
            slots,
            acquire_start: Cell::new(0),
        })
    }

    /// Try to acquire a slot from the arena without locking.
    ///
    /// Looks for a slot with:
    /// - `capacity >= rounded(requested)` -- the buffer is large enough
    /// - `Arc::strong_count(&buffer) == 1` -- only the arena holds a reference
    ///   (the slot is not in use by any live Tensor)
    ///
    /// On success: increments strong_count (returns Arc::clone),
    /// updates the round-robin pointer.
    ///
    /// On failure: returns `None` -> the caller must fall back to the pool.
    ///
    /// # Safety
    ///
    /// The caller MUST guarantee a GPU fence (`wait_until_completed_fast`)
    /// before a slot can be handed out for a new use.
    /// Otherwise the GPU may write to the buffer concurrently with a new kernel.
    pub fn try_acquire(&self, requested: usize) -> Option<Arc<Buffer>> {
        // Round-up to a power of two (like buf_size() in device.rs).
        // .max(64) -- minimum MTLBuffer alignment.
        let rounded = requested.saturating_sub(1).next_power_of_two().max(64);
        let n = self.slots.len();
        let start = self.acquire_start.get();

        for offset in 0..n {
            let idx = (start + offset) % n;
            let slot = &self.slots[idx];
            // Arc::strong_count -- AtomicUsize::load(Relaxed). Cheap.
            // Single-threaded inference: no TOCTOU race.
            if slot.capacity >= rounded && Arc::strong_count(&slot.buffer) == 1 {
                // Advance start so the next request does not start from the same slot.
                self.acquire_start.set((idx + 1) % n);
                return Some(Arc::clone(&slot.buffer));
            }
        }
        None
    }

    /// Total byte size of all pre-allocated slots.
    pub fn total_bytes(&self) -> usize {
        self.slots.iter().map(|s| s.capacity).sum()
    }

    /// Number of free slots (strong_count == 1).
    ///
    /// Useful for diagnostics: if `free_count() == 0` during forward --
    /// the arena is exhausted and the fallback to the pool is active.
    pub fn free_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| Arc::strong_count(&s.buffer) == 1)
            .count()
    }

    /// Capacity histogram: `(capacity_bytes, count)` sorted by capacity.
    pub fn capacity_histogram(&self) -> Vec<(usize, usize)> {
        let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
        for s in &self.slots {
            *hist.entry(s.capacity).or_insert(0) += 1;
        }
        hist.into_iter().collect()
    }
}

// ============================================================
// UnifiedScratchArena -- Phase 3c: one big MTLBuffer +
// bump allocator + offset dispatch.
//
// Design: a single MTLBuffer (~700 MB) is created at model load.
// All intermediate allocations are offsets into it.
// After each sync-fence (per N layers) the offset is reset,
// releasing all previous virtual allocations.
//
// This is a transliteration of a simplified ggml_dyn_tallocr.
// No complex free-list is needed -- the sync-per-N-layers boundary
// gives a natural reset point.
//
// # Usage
//
// 1. Create once at model load:
//    let arena = Arc::new(UnifiedScratchArena::new(&device, 764 * 1024 * 1024)?);
//
// 2. In the forward prefill loop, before each group of layers:
//    arena.reset(); // reset the bump pointer
//
// 3. On allocate_buffer -- instead of a new MTLBuffer:
//    let (buf, offset) = arena.try_acquire(size)?;
//    call_kernel(..., &buf, offset);
//
// 4. After the sync fence (wait_until_completed_fast):
//    // GPU is done, the offset is safe to reset.
//    arena.reset();
//
// # Safety
//
// The GPU MUST finish using all offsets from the current
// "episode" (the period between two reset() calls) BEFORE reset() is called.
// The caller ensures this via wait_until_completed_fast().
// ============================================================

/// The result of a successful acquire from UnifiedScratchArena.
pub struct UnifiedAlloc {
    /// Shared backing buffer (Arc -- the arena holds one more reference).
    pub buffer: Arc<Buffer>,
    /// Byte offset into the buffer.
    pub offset_in_bytes: usize,
}

/// Bump-allocator scratch arena: one MTLBuffer, offset dispatch.
///
/// Thread-safety: `Send + Sync` unsafe for the same reason as `ScratchArena`.
/// Inference is single-threaded; bump_offset is AtomicUsize for correctness
/// in a potential multi-threaded case (forward is not parallel right now).
pub struct UnifiedScratchArena {
    /// The single large MTLBuffer (StorageModeShared).
    backing: Arc<Buffer>,
    /// Current bump offset in bytes.
    bump_offset: AtomicUsize,
    /// Maximum size of the backing buffer.
    capacity: usize,
    /// Counter of exhausted allocations (diagnostics).
    exhausted_count: AtomicUsize,
}

unsafe impl Send for UnifiedScratchArena {}
unsafe impl Sync for UnifiedScratchArena {}

/// Metal requires a minimum alignment of 256 bytes for buffer offsets.
const METAL_BUFFER_OFFSET_ALIGNMENT: usize = 256;

#[inline(always)]
fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

impl UnifiedScratchArena {
    /// Create a UnifiedScratchArena with the given capacity (bytes).
    ///
    /// The whole backing buffer is allocated up front. Recommended size
    /// for Qwen3.5-2B pp4096: 764 MB (from Phase 1 profiling).
    pub fn new(device: &Device, capacity: usize) -> Result<Self, MetalKernelError> {
        let backing = device.new_buffer(capacity, MTLResourceOptions::StorageModeShared)?;
        Ok(Self {
            backing: Arc::new(backing),
            bump_offset: AtomicUsize::new(0),
            capacity,
            exhausted_count: AtomicUsize::new(0),
        })
    }

    /// Try to allocate `size` bytes from the arena.
    ///
    /// Returns `Some((Arc<Buffer>, offset))` if there is room.
    /// Returns `None` if the arena is exhausted (caller falls back to the pool).
    ///
    /// The offset is aligned to `METAL_BUFFER_OFFSET_ALIGNMENT` (256 bytes).
    pub fn try_acquire(&self, size: usize) -> Option<UnifiedAlloc> {
        if size == 0 {
            return None;
        }
        // Align size UP to a multiple of alignment for the next alloc.
        let aligned = align_up(size, METAL_BUFFER_OFFSET_ALIGNMENT);
        let offset = self.bump_offset.fetch_add(aligned, Ordering::Relaxed);
        if offset + aligned > self.capacity {
            // Roll back (arena exhausted, do not waste space further).
            self.bump_offset.fetch_sub(aligned, Ordering::Relaxed);
            self.exhausted_count.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(UnifiedAlloc {
            buffer: Arc::clone(&self.backing),
            offset_in_bytes: offset,
        })
    }

    /// Reset the bump offset to zero.
    ///
    /// DANGEROUS: call ONLY after `wait_until_completed_fast()`.
    /// The GPU must have finished all operations using previous offsets.
    #[inline]
    pub fn reset(&self) {
        self.bump_offset.store(0, Ordering::Relaxed);
    }

    /// Current bump usage in bytes.
    pub fn used_bytes(&self) -> usize {
        self.bump_offset.load(Ordering::Relaxed)
    }

    /// Backing buffer capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// How many times the arena has been exhausted since creation (diagnostics).
    pub fn exhausted_count(&self) -> usize {
        self.exhausted_count.load(Ordering::Relaxed)
    }
}