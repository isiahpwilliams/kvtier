//! Fixed-size block storage over one anonymous mapping. Uniform block size
//! makes allocation a `Vec::pop` and freeing a `Vec::push`.
//!
//! An mmap rather than a `Vec<u8>` because the address range never moves
//! (the server writes these addresses straight to a socket, Phase 4 hands
//! them to `cudaHostRegister`), pages commit lazily, and swapping `map_anon`
//! for a file-backed mapping gives us the NVMe tier.
//!
//! The mapping lives behind an `Arc` so a reader can hold on to one block
//! while the allocator keeps handing out others. What makes that sound is a
//! single invariant, enforced by `Slab` and `Index` together:
//!
//!   **A block is written once, before it is reachable, and never again.**
//!
//! `admit` writes a slot while it is still absent from the index, so no
//! reader can name it. Once published, the bytes are immutable until the
//! slot is freed, and a slot with pins outstanding cannot be freed.

use std::io;
use std::sync::Arc;

use memmap2::MmapMut;

use crate::block::BlockHash;

/// Index of a block-sized slot within the slab.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SlotId(u32);

impl SlotId {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[cfg(test)]
    pub(crate) fn for_tests(index: u32) -> Self {
        Self(index)
    }
}

/// The mapping itself, with no notion of which slots are in use.
pub struct SlabMemory {
    base: *mut u8,
    block_bytes: usize,
    capacity: usize,
    /// Keeps the mapping alive. The base address is captured once and stays
    /// valid regardless of where this struct is moved.
    _map: MmapMut,
}

// SAFETY: the raw pointer is a stable mmap base. Callers of `block` and
// `block_mut` uphold disjointness; see their safety contracts.
unsafe impl Send for SlabMemory {}
unsafe impl Sync for SlabMemory {}

impl SlabMemory {
    fn new(block_bytes: usize, capacity: usize) -> io::Result<Self> {
        let mut map = MmapMut::map_anon(block_bytes * capacity)?;
        Ok(Self {
            base: map.as_mut_ptr(),
            block_bytes,
            capacity,
            _map: map,
        })
    }

    fn offset(&self, slot: SlotId) -> *mut u8 {
        debug_assert!(slot.index() < self.capacity, "slot out of range");
        // SAFETY: the slot index is within capacity, so the result is inside
        // the mapping.
        unsafe { self.base.add(slot.index() * self.block_bytes) }
    }

    /// # Safety
    /// `slot` must be allocated, and no `&mut` to it may exist while the
    /// returned slice lives.
    unsafe fn block(&self, slot: SlotId) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.offset(slot), self.block_bytes) }
    }

    /// # Safety
    /// `slot` must be allocated and not yet reachable by any reader, and no
    /// other reference to it may exist.
    #[allow(clippy::mut_from_ref)]
    unsafe fn block_mut(&self, slot: SlotId) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.offset(slot), self.block_bytes) }
    }
}

/// A block held open for reading. Its slot cannot be freed or rewritten
/// while this exists, so the bytes stay valid without holding the store lock.
pub struct PinnedBlock {
    memory: Arc<SlabMemory>,
    hash: BlockHash,
    slot: SlotId,
}

impl PinnedBlock {
    pub fn hash(&self) -> BlockHash {
        self.hash
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: this block's index entry holds a pin, so its slot is not in
        // the free list and cannot be handed to `admit`. A published block is
        // never written again, so no `&mut` to it can exist.
        unsafe { self.memory.block(self.slot) }
    }
}

/// A slot reserved but not yet published in the index, so nothing can read
/// it. That is what makes it safe to fill outside the store lock.
pub struct BlockWriter {
    memory: Arc<SlabMemory>,
    slot: SlotId,
}

impl BlockWriter {
    pub fn slot(&self) -> SlotId {
        self.slot
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the slot is allocated and absent from the index, so no
        // reader can name it, and `&mut self` rules out a second writer.
        unsafe { self.memory.block_mut(self.slot) }
    }
}

pub struct Slab {
    memory: Arc<SlabMemory>,
    /// Free slots, LIFO: the last-freed slot is still cache-hot.
    free: Vec<u32>,
    live: usize,
}

impl Slab {
    pub fn new(block_bytes: usize, capacity: usize) -> io::Result<Self> {
        assert!(block_bytes > 0, "block_bytes must be non-zero");
        assert!(capacity > 0, "capacity must be non-zero");
        assert!(
            capacity <= u32::MAX as usize,
            "capacity exceeds SlotId range"
        );

        // Reversed so the first allocations hand out slot 0, 1, 2, ...
        let free = (0..capacity as u32).rev().collect();
        Ok(Self {
            memory: Arc::new(SlabMemory::new(block_bytes, capacity)?),
            free,
            live: 0,
        })
    }

    /// Claim a slot, or `None` when full. Never evicts: policy belongs to
    /// the layer above.
    pub fn alloc(&mut self) -> Option<SlotId> {
        let slot = self.free.pop()?;
        self.live += 1;
        Some(SlotId(slot))
    }

    pub fn free(&mut self, slot: SlotId) {
        debug_assert!(slot.index() < self.memory.capacity, "slot out of range");
        debug_assert!(!self.free.contains(&slot.0), "double free of {slot:?}");
        self.free.push(slot.0);
        self.live -= 1;
    }

    pub fn block(&self, slot: SlotId) -> &[u8] {
        // SAFETY: `&self` rules out a concurrent `block_mut`, which needs
        // `&mut self`.
        unsafe { self.memory.block(slot) }
    }

    pub fn block_mut(&mut self, slot: SlotId) -> &mut [u8] {
        // SAFETY: `&mut self` rules out every other reference through this
        // `Slab`. Callers must not have published the slot yet, or a
        // `PinnedBlock` could be reading it.
        unsafe { self.memory.block_mut(slot) }
    }

    /// Hand out a reserved slot for filling outside the lock. The caller must
    /// not publish `slot` in the index until the writer is done with it.
    pub fn writer(&self, slot: SlotId) -> BlockWriter {
        BlockWriter {
            memory: Arc::clone(&self.memory),
            slot,
        }
    }

    /// Hold a slot open for reading past the lifetime of a `&self` borrow.
    /// The caller must have pinned `hash` in the index first.
    pub fn pinned(&self, hash: BlockHash, slot: SlotId) -> PinnedBlock {
        PinnedBlock {
            memory: Arc::clone(&self.memory),
            hash,
            slot,
        }
    }

    pub fn block_bytes(&self) -> usize {
        self.memory.block_bytes
    }

    pub fn capacity(&self) -> usize {
        self.memory.capacity
    }

    /// Slots currently handed out.
    pub fn live(&self) -> usize {
        self.live
    }

    pub fn is_full(&self) -> bool {
        self.free.is_empty()
    }

    /// Reserved address space, not resident memory.
    pub fn reserved_bytes(&self) -> usize {
        self.block_bytes() * self.capacity()
    }

    pub fn utilization(&self) -> f64 {
        self.live as f64 / self.capacity() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_until_full_then_recycle() {
        let mut slab = Slab::new(64, 3).unwrap();
        let a = slab.alloc().unwrap();
        let b = slab.alloc().unwrap();
        let c = slab.alloc().unwrap();
        assert_eq!(slab.live(), 3);
        assert!(slab.is_full());
        assert!(slab.alloc().is_none(), "must refuse rather than overrun");

        slab.free(b);
        assert_eq!(slab.live(), 2);
        assert_eq!(slab.alloc(), Some(b), "LIFO reuse keeps the hot slot hot");
        assert_ne!(a, c);
    }

    #[test]
    fn slots_do_not_alias() {
        let mut slab = Slab::new(64, 4).unwrap();
        let a = slab.alloc().unwrap();
        let b = slab.alloc().unwrap();

        slab.block_mut(a).fill(0xAA);
        slab.block_mut(b).fill(0xBB);

        assert!(slab.block(a).iter().all(|&x| x == 0xAA));
        assert!(slab.block(b).iter().all(|&x| x == 0xBB));
    }

    #[test]
    fn fresh_mapping_is_zeroed() {
        let mut slab = Slab::new(64, 2).unwrap();
        let slot = slab.alloc().unwrap();
        assert!(slab.block(slot).iter().all(|&x| x == 0));
    }

    #[test]
    fn large_reservation_is_lazy() {
        // 16 GiB of address space, a few KiB of it actually touched.
        let mut slab = Slab::new(2 * 1024 * 1024, 8192).unwrap();
        assert_eq!(slab.reserved_bytes(), 16 * 1024 * 1024 * 1024);
        let slot = slab.alloc().unwrap();
        slab.block_mut(slot)[..4].copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(&slab.block(slot)[..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn a_pinned_block_outlives_the_borrow_and_ignores_other_writes() {
        let mut slab = Slab::new(64, 4).unwrap();
        let held = slab.alloc().unwrap();
        slab.block_mut(held).fill(0x11);

        let pinned = slab.pinned(BlockHash::from_bytes([0; 16]), held);

        // Writing other slots must not disturb the pinned one.
        for _ in 0..3 {
            let other = slab.alloc().unwrap();
            slab.block_mut(other).fill(0x22);
        }
        assert!(pinned.bytes().iter().all(|&x| x == 0x11));
    }
}
