//! The disk tier: the same fixed-slot discipline as the RAM slab, backed by a
//! file. Eviction from RAM becomes demotion here, so a block that leaves
//! memory costs a read on its next hit rather than a full prefill.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::block::BlockLayout;
use crate::evict::CostModel;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct DiskSlot(u32);

impl DiskSlot {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DiskStats {
    pub reads: u64,
    pub writes: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

/// When a demoted block is worth keeping.
///
/// The choice on eviction is three-way: hold in RAM, push to disk, or drop.
/// Dropping is right only when reading the block back would cost more than
/// asking the GPU to rebuild it -- which is a real crossover, since a block
/// is megabytes of bytes but only a few milliseconds of compute.
#[derive(Clone, Copy, Debug)]
pub struct TierCosts {
    pub disk_bandwidth: f64,
    pub disk_latency_secs: f64,
    /// GPU seconds to rebuild one block whose prefix is already cached, at
    /// depth zero. Scaled by the cost model for deeper blocks.
    pub recompute_secs: f64,
}

impl Default for TierCosts {
    fn default() -> Self {
        Self {
            // A mid-range NVMe drive.
            disk_bandwidth: 3.0e9,
            disk_latency_secs: 100e-6,
            // 16 tokens of Llama-3-8B prefill on an A100-class part.
            recompute_secs: 3.3e-3,
        }
    }
}

impl TierCosts {
    pub fn fetch_secs(&self, block_bytes: usize) -> f64 {
        self.disk_latency_secs + block_bytes as f64 / self.disk_bandwidth
    }

    /// True when a fetch beats a rebuild, so the block earns its disk slot.
    pub fn worth_demoting(
        &self,
        cost: &CostModel,
        layout: &BlockLayout,
        depth_tokens: u32,
    ) -> bool {
        self.fetch_secs(layout.block_bytes())
            < self.recompute_secs * cost.recompute_cost(depth_tokens)
    }
}

pub struct DiskTier {
    file: File,
    path: PathBuf,
    /// Unlinked at creation, so the backing file disappears with the process.
    unlinked: bool,
    block_bytes: usize,
    capacity: usize,
    free: Vec<u32>,
    live: usize,
    stats: DiskStats,
}

impl DiskTier {
    pub fn create(path: &Path, block_bytes: usize, capacity: usize) -> io::Result<Self> {
        assert!(block_bytes > 0 && capacity > 0);
        assert!(
            capacity <= u32::MAX as usize,
            "capacity exceeds DiskSlot range"
        );

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        // Size it up front so writes never extend the file mid-operation.
        file.set_len((block_bytes * capacity) as u64)?;

        Ok(Self {
            file,
            path: path.to_path_buf(),
            unlinked: false,
            block_bytes,
            capacity,
            free: (0..capacity as u32).rev().collect(),
            live: 0,
            stats: DiskStats::default(),
        })
    }

    /// A tier whose file is unlinked immediately: still readable through the
    /// open descriptor, gone the moment the process exits.
    pub fn temporary(block_bytes: usize, capacity: usize) -> io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "kvtier-{}-{unique}-{}.tier",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));

        let mut tier = Self::create(&path, block_bytes, capacity)?;
        std::fs::remove_file(&path)?;
        tier.unlinked = true;
        Ok(tier)
    }

    pub fn alloc(&mut self) -> Option<DiskSlot> {
        let slot = self.free.pop()?;
        self.live += 1;
        Some(DiskSlot(slot))
    }

    pub fn free(&mut self, slot: DiskSlot) {
        debug_assert!(slot.index() < self.capacity, "slot out of range");
        debug_assert!(!self.free.contains(&slot.0), "double free of {slot:?}");
        self.free.push(slot.0);
        self.live -= 1;
    }

    fn offset(&self, slot: DiskSlot) -> u64 {
        (slot.index() * self.block_bytes) as u64
    }

    pub fn write_block(&mut self, slot: DiskSlot, data: &[u8]) -> io::Result<()> {
        assert_eq!(data.len(), self.block_bytes, "payload must be one block");
        self.file.write_all_at(data, self.offset(slot))?;
        self.stats.writes += 1;
        self.stats.bytes_written += data.len() as u64;
        Ok(())
    }

    pub fn read_block(&mut self, slot: DiskSlot, into: &mut [u8]) -> io::Result<()> {
        assert_eq!(
            into.len(),
            self.block_bytes,
            "destination must be one block"
        );
        self.file.read_exact_at(into, self.offset(slot))?;
        self.stats.reads += 1;
        self.stats.bytes_read += into.len() as u64;
        Ok(())
    }

    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn live(&self) -> usize {
        self.live
    }

    pub fn is_full(&self) -> bool {
        self.free.is_empty()
    }

    pub fn stats(&self) -> DiskStats {
        self.stats
    }
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        if !self.unlinked {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_round_trip_through_the_file() {
        let mut tier = DiskTier::temporary(1024, 4).unwrap();
        let first = tier.alloc().unwrap();
        let second = tier.alloc().unwrap();

        tier.write_block(first, &[0xAA; 1024]).unwrap();
        tier.write_block(second, &[0xBB; 1024]).unwrap();

        let mut buffer = vec![0u8; 1024];
        tier.read_block(first, &mut buffer).unwrap();
        assert!(buffer.iter().all(|&b| b == 0xAA));
        tier.read_block(second, &mut buffer).unwrap();
        assert!(buffer.iter().all(|&b| b == 0xBB));
    }

    #[test]
    fn slots_run_out_and_recycle() {
        let mut tier = DiskTier::temporary(64, 2).unwrap();
        let first = tier.alloc().unwrap();
        tier.alloc().unwrap();
        assert!(tier.is_full());
        assert!(tier.alloc().is_none());

        tier.free(first);
        assert_eq!(tier.alloc(), Some(first));
    }

    #[test]
    fn a_deep_block_is_worth_keeping_and_a_free_one_is_not() {
        let costs = TierCosts::default();
        let cost = CostModel::default();
        let layout = BlockLayout::llama3_8b();

        // 2 MiB off NVMe is ~800us against ~3.3ms of prefill: keep it.
        assert!(costs.worth_demoting(&cost, &layout, 0));

        // If rebuilding were nearly free, the disk slot would not pay for
        // itself and the block should just be dropped.
        let cheap = TierCosts {
            recompute_secs: 10e-6,
            ..TierCosts::default()
        };
        assert!(!cheap.worth_demoting(&cost, &layout, 0));
    }
}
