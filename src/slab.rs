//! Fixed-size block storage over one anonymous mapping. Uniform block size
//! makes allocation a `Vec::pop` and freeing a `Vec::push`.
//!
//! An mmap rather than a `Vec<u8>` because the address range never moves
//! (Phase 2 hands these addresses to vectored writes, Phase 4 to
//! `cudaHostRegister`), pages commit lazily, and swapping `map_anon` for a
//! file-backed mapping gives us the NVMe tier.

use std::io;
use std::ops::Range;

use memmap2::MmapMut;

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

pub struct Slab {
    memory: MmapMut,
    block_bytes: usize,
    capacity: usize,
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

        let memory = MmapMut::map_anon(block_bytes * capacity)?;
        // Reversed so the first allocations hand out slot 0, 1, 2, ...
        let free = (0..capacity as u32).rev().collect();

        Ok(Self {
            memory,
            block_bytes,
            capacity,
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
        debug_assert!(slot.index() < self.capacity, "slot out of range");
        debug_assert!(!self.free.contains(&slot.0), "double free of {slot:?}");
        self.free.push(slot.0);
        self.live -= 1;
    }

    fn range(&self, slot: SlotId) -> Range<usize> {
        let start = slot.index() * self.block_bytes;
        start..start + self.block_bytes
    }

    pub fn block(&self, slot: SlotId) -> &[u8] {
        &self.memory[self.range(slot)]
    }

    pub fn block_mut(&mut self, slot: SlotId) -> &mut [u8] {
        let range = self.range(slot);
        &mut self.memory[range]
    }

    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    pub fn capacity(&self) -> usize {
        self.capacity
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
        self.block_bytes * self.capacity
    }

    pub fn utilization(&self) -> f64 {
        self.live as f64 / self.capacity as f64
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
}
