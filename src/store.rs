//! Name a sequence, find the longest resident prefix, admit what was
//! missing. The whole cache minus the network, the tiers and eviction.

use std::io;

use crate::block::{BlockHash, BlockLayout, PrefixHasher, TokenId};
use crate::evict::{CostModel, GreedyDual};
use crate::index::{Index, Place};
use crate::slab::{BlockWriter, PinnedBlock, Slab, SlotId};
use crate::tier::{DiskReader, DiskSlot, DiskStats, DiskTier, DiskWriter, TierCosts};

#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub lookups: u64,
    /// Blocks asked about across all lookups.
    pub queried_blocks: u64,
    /// Blocks that were resident when asked about.
    pub hit_blocks: u64,
    pub inserted_blocks: u64,
    /// Admits that found the block already present: each one is a prefill
    /// some other request had already paid for.
    pub deduped_blocks: u64,
    /// Admits refused: the slab was full and nothing was worth displacing.
    pub rejected_blocks: u64,
    /// Blocks removed from the cache entirely.
    pub evicted_blocks: u64,
    /// Blocks pushed from RAM to disk rather than dropped.
    pub demoted_blocks: u64,
    /// Demotions that had to write to disk with the store lock held. The
    /// number writeback exists to drive to zero.
    pub blocking_demotions: u64,
    /// Blocks copied to disk ahead of time by the writeback loop.
    pub written_back: u64,
    /// Blocks read back from disk on a hit.
    pub promoted_blocks: u64,
    pub bytes_admitted: u64,
}

impl Stats {
    pub fn hit_rate(&self) -> f64 {
        if self.queried_blocks == 0 {
            return 0.0;
        }
        self.hit_blocks as f64 / self.queried_blocks as f64
    }
}

/// Result of naming a sequence and probing for it.
#[derive(Debug, Clone)]
pub struct Lookup {
    /// Names of every full block in the sequence, in order.
    pub hashes: Vec<BlockHash>,
    /// How many leading blocks were resident.
    pub matched: usize,
    tokens_per_block: usize,
}

impl Lookup {
    /// Tokens the engine can skip prefilling.
    pub fn hit_tokens(&self) -> usize {
        self.matched * self.tokens_per_block
    }

    /// Blocks the engine must compute itself, and hand back afterwards.
    pub fn missing(&self) -> &[BlockHash] {
        &self.hashes[self.matched..]
    }

    pub fn is_full_hit(&self) -> bool {
        !self.hashes.is_empty() && self.matched == self.hashes.len()
    }
}

/// One block of a fetch: already in RAM, or waiting on a disk read.
pub enum FetchPart {
    Ready(PinnedBlock),
    Pending(Promotion),
}

/// A dirty RAM block with a disk slot reserved for its copy.
pub struct Writeback {
    hash: BlockHash,
    slot: SlotId,
    disk_slot: DiskSlot,
    block: PinnedBlock,
    written: bool,
}

impl Writeback {
    /// Copy the block to disk. Safe with the store lock released: the block
    /// is pinned so it cannot move, and published blocks are immutable, so
    /// the bytes cannot change under the write.
    pub fn flush(&mut self, writer: &DiskWriter) -> std::io::Result<()> {
        writer.write(self.disk_slot, self.block.bytes())?;
        self.written = true;
        Ok(())
    }
}

/// A demoted block with a RAM slot reserved for it, waiting to be filled.
pub struct Promotion {
    hash: BlockHash,
    disk_slot: DiskSlot,
    writer: BlockWriter,
    filled: bool,
}

impl Promotion {
    /// Read the block off disk. Safe to call with the store lock released:
    /// the destination slot is not in the index, so nothing can see it, and
    /// the block is pinned, so nothing can evict what we are reading.
    pub fn fill(&mut self, reader: &DiskReader) -> std::io::Result<()> {
        reader.read(self.disk_slot, self.writer.bytes_mut())?;
        self.filled = true;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Admit {
    Inserted,
    /// Some other sequence already computed this block.
    AlreadyPresent,
    /// Full, and every resident block is pinned or has children.
    OutOfSpace,
    /// The preceding block is not resident, so this one is unusable.
    OrphanParent,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AdmitReport {
    pub inserted: usize,
    pub deduped: usize,
    /// Declined, including everything after the first failure: once a link
    /// is missing the rest of the chain is unusable.
    pub dropped: usize,
}

pub struct KvStore {
    model_id: String,
    layout: BlockLayout,
    hasher: PrefixHasher,
    slab: Slab,
    index: Index,
    disk: Option<DiskTier>,
    /// One clock per tier: a block's standing in RAM says nothing about its
    /// standing among the blocks already on disk.
    ram_policy: GreedyDual,
    disk_policy: GreedyDual,
    tier_costs: TierCosts,
    cost: CostModel,
    stats: Stats,
}

impl KvStore {
    pub fn new(model_id: &str, layout: BlockLayout, capacity_blocks: usize) -> io::Result<Self> {
        let hasher = PrefixHasher::new(model_id, &layout);
        let slab = Slab::new(layout.block_bytes(), capacity_blocks)?;
        Ok(Self {
            model_id: model_id.to_owned(),
            layout,
            hasher,
            slab,
            index: Index::new(),
            disk: None,
            ram_policy: GreedyDual::default(),
            disk_policy: GreedyDual::default(),
            tier_costs: TierCosts::default(),
            cost: CostModel::default(),
            stats: Stats::default(),
        })
    }

    /// Replace the recompute cost model, which is per-model.
    pub fn with_cost_model(mut self, cost: CostModel) -> Self {
        self.cost = cost;
        self.ram_policy = GreedyDual::new(cost);
        self.disk_policy = GreedyDual::new(cost);
        self
    }

    /// Attach a disk tier. Without one, eviction means deletion.
    pub fn with_disk_tier(mut self, tier: DiskTier) -> Self {
        assert_eq!(
            tier.block_bytes(),
            self.layout.block_bytes(),
            "disk tier block size must match the layout"
        );
        self.disk = Some(tier);
        self
    }

    pub fn with_tier_costs(mut self, costs: TierCosts) -> Self {
        self.tier_costs = costs;
        self
    }

    pub fn disk_stats(&self) -> Option<DiskStats> {
        self.disk.as_ref().map(DiskTier::stats)
    }

    pub fn disk_blocks(&self) -> usize {
        self.disk.as_ref().map_or(0, DiskTier::live)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Read-only view of the index, for tests and inspection.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Blocks the policy is currently tracking as eviction candidates,
    /// including stale entries awaiting compaction.
    pub fn eviction_candidates(&self) -> usize {
        self.ram_policy.candidates() + self.disk_policy.candidates()
    }

    /// Slots in the RAM tier, whether or not they are in use.
    pub fn ram_capacity(&self) -> usize {
        self.slab.capacity()
    }

    pub fn resident_blocks(&self) -> usize {
        self.index.len()
    }

    pub fn utilization(&self) -> f64 {
        self.slab.utilization()
    }

    /// Name every full block of `tokens` and report how many are resident.
    pub fn lookup(&mut self, tokens: &[TokenId]) -> Lookup {
        let hashes = self.hasher.chain(tokens);
        let matched = self.match_prefix(&hashes);

        Lookup {
            hashes,
            matched,
            tokens_per_block: self.layout.tokens_per_block,
        }
    }

    /// Probe names computed elsewhere. Phase 2's wire carries hashes rather
    /// than tokens, so a peer never has to ship us a whole prompt to ask.
    pub fn match_prefix(&mut self, hashes: &[BlockHash]) -> usize {
        let matched = self.index.match_prefix(hashes);
        for &hash in &hashes[..matched] {
            self.reprice(hash);
        }

        self.stats.lookups += 1;
        self.stats.queried_blocks += hashes.len() as u64;
        self.stats.hit_blocks += matched as u64;

        matched
    }

    /// Raise a block's eviction priority to reflect having just been used,
    /// and re-offer it to the policy at its new value.
    fn reprice(&mut self, hash: BlockHash) {
        let Some(entry) = self.index.get(hash).copied() else {
            return;
        };
        let in_ram = entry.place.is_ram();
        let policy = if in_ram {
            &mut self.ram_policy
        } else {
            &mut self.disk_policy
        };

        let priority = policy.priority_for(entry.depth_tokens);
        policy.offer(hash, priority, entry.last_access);
        let crowded = policy.should_compact(self.index.len());
        // Stamps `priority_access` from `last_access`, matching what was
        // just offered.
        self.index.set_priority(hash, priority);

        if crowded {
            let index = &self.index;
            if in_ram {
                self.ram_policy.compact(index, |e| e.place.is_ram());
            } else {
                self.disk_policy.compact(index, |e| !e.place.is_ram());
            }
        }
    }

    /// Borrow a block's bytes, if it is in RAM. A demoted block reads as
    /// `None` here; `pin_run` promotes it first.
    pub fn read(&self, hash: BlockHash) -> Option<&[u8]> {
        let slot = self.index.get(hash)?.place.ram()?;
        Some(self.slab.block(slot))
    }

    /// Whether the cache holds this block at all, in either tier.
    pub fn contains(&self, hash: BlockHash) -> bool {
        self.index.contains(hash)
    }

    /// Hold the resident leading run of `hashes` open for reading.
    ///
    /// The returned guards outlive this borrow, so the server can drop the
    /// store lock and spend the whole socket write unlocked. Every guard must
    /// come back to `unpin_all`, including on an error path -- a lost pin
    /// makes its block permanently unevictable.
    pub fn pin_run(&mut self, hashes: &[BlockHash]) -> Vec<PinnedBlock> {
        let mut pinned = Vec::with_capacity(hashes.len());
        for &hash in hashes {
            // A demoted block has to come back to RAM before its bytes can be
            // borrowed. Failing that, the run ends here.
            if self.index.get(hash).is_some_and(|e| !e.place.is_ram())
                && self.promote(hash).is_err()
            {
                break;
            }
            match self.index.pin(hash).and_then(Place::ram) {
                Some(slot) => {
                    pinned.push(self.slab.pinned(hash, slot));
                    self.reprice(hash);
                }
                // A gap ends the run: nothing past it is usable anyway.
                None => break,
            }
        }

        self.stats.lookups += 1;
        self.stats.queried_blocks += hashes.len() as u64;
        self.stats.hit_blocks += pinned.len() as u64;
        pinned
    }

    /// Reserve the resident leading run without doing any I/O.
    ///
    /// Blocks already in RAM come back ready; demoted ones come back as
    /// `Pending`, pinned and holding a reserved RAM slot, for the caller to
    /// fill off the lock. Pair every call with `finish_fetch`, or the pins
    /// and slots reserved here leak.
    pub fn begin_fetch(&mut self, hashes: &[BlockHash]) -> Vec<FetchPart> {
        let mut parts: Vec<FetchPart> = Vec::with_capacity(hashes.len());

        for &hash in hashes {
            let Some(place) = self.index.pin(hash) else {
                break; // a gap ends the run
            };
            self.reprice(hash);

            match place {
                Place::Ram { slot, .. } => {
                    parts.push(FetchPart::Ready(self.slab.pinned(hash, slot)))
                }
                Place::Disk(disk_slot) => {
                    // The pin above keeps this block off both eviction heaps
                    // while we are reading it.
                    let Some(slot) = self.reserve_slot(hash) else {
                        self.index.unpin(hash);
                        break;
                    };
                    parts.push(FetchPart::Pending(Promotion {
                        hash,
                        disk_slot,
                        writer: self.slab.writer(slot),
                        filled: false,
                    }));
                }
            }
        }

        self.stats.lookups += 1;
        self.stats.queried_blocks += hashes.len() as u64;
        self.stats.hit_blocks += parts.len() as u64;
        parts
    }

    fn reserve_slot(&mut self, protect: BlockHash) -> Option<SlotId> {
        match self.slab.alloc() {
            Some(slot) => Some(slot),
            None => match self.make_room(Some(protect)) {
                Ok(()) => self.slab.alloc(),
                Err(_) => None,
            },
        }
    }

    /// Publish whatever the caller managed to fill, and hand back the run.
    ///
    /// Truncates at the first unfilled block: a run with a hole in it is not
    /// a run, and the engine cannot use anything past the gap.
    pub fn finish_fetch(&mut self, parts: Vec<FetchPart>) -> Vec<PinnedBlock> {
        let mut pinned = Vec::with_capacity(parts.len());
        let mut truncated = false;
        let mut reads = 0u64;
        let mut bytes = 0u64;

        for part in parts {
            match part {
                _ if truncated => match part {
                    FetchPart::Ready(block) => self.index.unpin(block.hash()),
                    FetchPart::Pending(promotion) => self.abandon(promotion),
                },
                FetchPart::Ready(block) => pinned.push(block),
                FetchPart::Pending(promotion) if promotion.filled => {
                    reads += 1;
                    bytes += self.layout.block_bytes() as u64;
                    pinned.push(self.publish(promotion));
                }
                FetchPart::Pending(promotion) => {
                    self.abandon(promotion);
                    truncated = true;
                }
            }
        }

        if let Some(disk) = self.disk.as_mut() {
            disk.note_reads(reads, bytes);
        }
        self.stats.promoted_blocks += reads;
        pinned
    }

    /// A filled promotion becomes the block's new home.
    fn publish(&mut self, promotion: Promotion) -> PinnedBlock {
        let slot = promotion.writer.slot();
        // Keep the disk copy: the block is instantly clean, so if it gets
        // pushed back out there is nothing to write.
        self.index.set_place(
            promotion.hash,
            Place::Ram {
                slot,
                backing: Some(promotion.disk_slot),
            },
        );
        self.reprice(promotion.hash);
        self.slab.pinned(promotion.hash, slot)
    }

    /// Give back what a promotion reserved. The block stays on disk.
    fn abandon(&mut self, promotion: Promotion) {
        self.slab.free(promotion.writer.slot());
        self.index.unpin(promotion.hash);
    }

    pub fn disk_reader(&self) -> Option<DiskReader> {
        self.disk.as_ref().map(DiskTier::reader)
    }

    pub fn disk_writer(&self) -> Option<DiskWriter> {
        self.disk.as_ref().map(DiskTier::writer)
    }

    /// RAM blocks with no disk copy, whose slots cannot be reclaimed without
    /// a write first.
    pub fn dirty_blocks(&self) -> usize {
        self.index
            .iter()
            .filter(|(_, entry)| entry.place.is_dirty())
            .count()
    }

    /// Pick up to `max` dirty blocks to copy to disk, cheapest-to-keep first.
    ///
    /// Takes the blocks the eviction policy would give up next, so the work
    /// lands exactly where it will be needed. Each job holds a pin and a
    /// reserved disk slot; pass them all to `finish_writeback` or both leak.
    pub fn begin_writeback(&mut self, max: usize) -> Vec<Writeback> {
        if self.disk.is_none() || max == 0 {
            return Vec::new();
        }

        let candidates = self
            .ram_policy
            .peek_victims(&self.index, max, |entry| entry.place.is_dirty());

        let mut jobs = Vec::new();
        for hash in candidates {
            let Some(entry) = self.index.get(hash).copied() else {
                continue;
            };
            // A block we would rather drop than demote is not worth a write.
            if !self.worth_demoting(entry.depth_tokens) {
                continue;
            }
            let Some(disk_slot) = self.disk.as_mut().and_then(DiskTier::alloc) else {
                break; // disk is full; the blocking path can still make room
            };
            let Some(slot) = self.index.pin(hash).and_then(Place::ram) else {
                self.disk.as_mut().expect("checked above").free(disk_slot);
                continue;
            };

            jobs.push(Writeback {
                hash,
                slot,
                disk_slot,
                block: self.slab.pinned(hash, slot),
                written: false,
            });
        }
        jobs
    }

    /// Record the copies that landed, making those blocks free to evict.
    pub fn finish_writeback(&mut self, jobs: Vec<Writeback>) {
        let mut written = 0u64;
        let mut bytes = 0u64;

        for job in jobs {
            if job.written {
                // Only if it is still where we left it: a concurrent demotion
                // may have moved it while the write was in flight.
                if self.index.get(job.hash).is_some_and(|e| e.place.is_dirty()) {
                    self.index.set_place(
                        job.hash,
                        Place::Ram {
                            slot: job.slot,
                            backing: Some(job.disk_slot),
                        },
                    );
                    written += 1;
                    bytes += self.layout.block_bytes() as u64;
                } else if let Some(disk) = self.disk.as_mut() {
                    disk.free(job.disk_slot);
                }
            } else if let Some(disk) = self.disk.as_mut() {
                disk.free(job.disk_slot);
            }
            self.index.unpin(job.hash);
        }

        if let Some(disk) = self.disk.as_mut() {
            disk.note_writes(written, bytes);
        }
        self.stats.written_back += written;
    }

    pub fn unpin_all(&mut self, blocks: &[PinnedBlock]) {
        for block in blocks {
            self.index.unpin(block.hash());
        }
    }

    /// Admit one block. `parent` is the preceding block's name, or `None`
    /// for the first block of a sequence.
    pub fn admit(
        &mut self,
        hash: BlockHash,
        parent: Option<BlockHash>,
        depth_tokens: u32,
        data: &[u8],
    ) -> Admit {
        assert_eq!(
            data.len(),
            self.slab.block_bytes(),
            "payload must be exactly one block under this layout"
        );

        if self.index.contains(hash) {
            self.stats.deduped_blocks += 1;
            return Admit::AlreadyPresent;
        }
        if let Some(parent) = parent
            && !self.index.contains(parent)
        {
            return Admit::OrphanParent;
        }

        // Allocate after the checks so a rejected admit has nothing to undo.
        let slot = match self.slab.alloc() {
            Some(slot) => slot,
            None => match self.make_room(parent) {
                Ok(()) => self.slab.alloc().expect("eviction freed a slot"),
                Err(refusal) => {
                    self.stats.rejected_blocks += 1;
                    return refusal;
                }
            },
        };

        self.slab.block_mut(slot).copy_from_slice(data);
        self.index
            .insert(hash, parent, Place::dirty(slot), depth_tokens)
            .expect("residency and duplication were checked above");
        self.reprice(hash);

        self.stats.inserted_blocks += 1;
        self.stats.bytes_admitted += data.len() as u64;
        Admit::Inserted
    }

    /// Free one slot for a block of this depth, or say why we would not.
    ///
    /// `parent` is excluded from selection. It is a leaf until the block we
    /// are admitting lands, which makes it the most attractive victim in the
    /// store -- and evicting it would orphan the very block we came to insert.
    fn make_room(&mut self, parent: Option<BlockHash>) -> Result<(), Admit> {
        // Always admit once a victim exists, even when the newcomer models
        // as cheaper. Refusing on cost ossifies the cache: new blocks start
        // shallow and lose to the deep tails already resident, so nothing is
        // admitted, nothing evicted, and the clock never rises to break the
        // tie.
        loop {
            let Some((victim, priority)) =
                self.ram_policy
                    .select_victim(&self.index, parent, |e| e.place.is_ram())
            else {
                return Err(Admit::OutOfSpace);
            };
            let entry = *self.index.get(victim).expect("selected from the index");

            // A clean block costs nothing to give up: its bytes are already
            // on disk, so we just drop the RAM slot.
            if let Place::Ram {
                slot,
                backing: Some(backing),
            } = entry.place
            {
                self.release_clean(victim, slot, backing);
                self.ram_policy.note_eviction(priority);
                return Ok(());
            }

            // Otherwise pay for the write here. Demotion beats dropping
            // whenever a read back is cheaper than a rebuild, and unlike
            // dropping it works on any block: a demoted parent is still
            // there for its children.
            if self.worth_demoting(entry.depth_tokens) && self.demote(victim).is_ok() {
                self.ram_policy.note_eviction(priority);
                return Ok(());
            }
            if entry.children == 0 {
                self.drop_block(victim, priority);
                return Ok(());
            }
            // An internal block we could not demote is not removable either.
            // It leaves the heap until its next access or its last child goes.
        }
    }

    fn worth_demoting(&self, depth_tokens: u32) -> bool {
        self.disk.is_some()
            && self
                .tier_costs
                .worth_demoting(&self.cost, &self.layout, depth_tokens)
    }

    /// Move a block's bytes to disk and free its RAM slot.
    fn demote(&mut self, hash: BlockHash) -> Result<(), ()> {
        let Some(slot) = self.index.get(hash).and_then(|e| e.place.ram()) else {
            return Err(());
        };
        if self.disk.as_ref().is_some_and(DiskTier::is_full) {
            self.make_disk_room()?;
        }

        let disk = self.disk.as_mut().ok_or(())?;
        let disk_slot = disk.alloc().ok_or(())?;
        if disk.write_block(disk_slot, self.slab.block(slot)).is_err() {
            self.disk.as_mut().expect("checked above").free(disk_slot);
            return Err(());
        }

        self.slab.free(slot);
        self.index.set_place(hash, Place::Disk(disk_slot));
        self.stats.demoted_blocks += 1;
        self.stats.blocking_demotions += 1;
        self.reprice(hash);
        Ok(())
    }

    /// Reclaim a clean block's RAM slot. Its bytes are already on disk, so
    /// this is bookkeeping, not I/O -- which is the whole point of writeback.
    fn release_clean(&mut self, hash: BlockHash, slot: SlotId, backing: DiskSlot) {
        self.slab.free(slot);
        self.index.set_place(hash, Place::Disk(backing));
        self.stats.demoted_blocks += 1;
        self.reprice(hash);
    }

    /// Read a block back into RAM so its bytes can be borrowed.
    fn promote(&mut self, hash: BlockHash) -> Result<(), ()> {
        let Some(disk_slot) = self.index.get(hash).and_then(|e| e.place.disk()) else {
            return Err(());
        };

        let slot = match self.slab.alloc() {
            Some(slot) => slot,
            // Protect the block we are promoting: it is on disk, so demoting
            // it again would be a pointless round trip.
            None => match self.make_room(Some(hash)) {
                Ok(()) => self.slab.alloc().expect("eviction freed a slot"),
                Err(_) => return Err(()),
            },
        };

        let disk = self.disk.as_mut().ok_or(())?;
        if disk
            .read_block(disk_slot, self.slab.block_mut(slot))
            .is_err()
        {
            self.slab.free(slot);
            return Err(());
        }
        // Keep the disk copy, as the async path does: the block is clean the
        // moment it lands, so pushing it back out costs nothing.
        self.index.set_place(
            hash,
            Place::Ram {
                slot,
                backing: Some(disk_slot),
            },
        );
        self.stats.promoted_blocks += 1;
        self.reprice(hash);
        Ok(())
    }

    /// Drop the cheapest disk-resident leaf, to make room for a demotion.
    fn make_disk_room(&mut self) -> Result<(), ()> {
        let (victim, priority) = self
            .disk_policy
            .select_victim(&self.index, None, |e| !e.place.is_ram() && e.children == 0)
            .ok_or(())?;
        self.disk_policy.note_eviction(priority);
        self.remove_block(victim);
        Ok(())
    }

    fn drop_block(&mut self, victim: BlockHash, priority: f64) {
        self.ram_policy.note_eviction(priority);
        self.remove_block(victim);
    }

    fn remove_block(&mut self, victim: BlockHash) {
        let entry = self
            .index
            .remove(victim)
            .expect("the policy only offers removable blocks");
        if let Some(slot) = entry.place.ram() {
            self.slab.free(slot);
        }
        if let Some(backing) = entry.place.backing() {
            self.disk
                .as_mut()
                .expect("a disk copy needs a tier")
                .free(backing);
        }
        self.stats.evicted_blocks += 1;

        // The parent may have just become a leaf, which makes it removable.
        if let Some(parent) = entry.parent
            && self.index.get(parent).is_some_and(|e| e.children == 0)
        {
            self.reprice(parent);
        }
    }

    /// Admit the KV a request just computed: `blocks[i]` is the payload for
    /// the i-th full block of `tokens`. The shape the engine connector calls.
    pub fn admit_sequence(&mut self, tokens: &[TokenId], blocks: &[&[u8]]) -> AdmitReport {
        let hashes = self.hasher.chain(tokens);
        assert!(
            blocks.len() <= hashes.len(),
            "got {} payloads for {} full blocks",
            blocks.len(),
            hashes.len()
        );

        let names: Vec<(BlockHash, u32)> = hashes
            .iter()
            .take(blocks.len())
            .enumerate()
            .map(|(i, &hash)| (hash, ((i + 1) * self.layout.tokens_per_block) as u32))
            .collect();

        self.admit_chain(None, &names, blocks)
    }

    /// Admit a run of blocks named elsewhere. `parent` is what the first
    /// block attaches to; after that each block's parent is the one before
    /// it. This is the wire's admit path -- a peer sends names and bytes,
    /// never tokens.
    pub fn admit_chain(
        &mut self,
        parent: Option<BlockHash>,
        names: &[(BlockHash, u32)],
        blocks: &[&[u8]],
    ) -> AdmitReport {
        assert_eq!(names.len(), blocks.len(), "one name per payload");

        let mut report = AdmitReport::default();
        let mut parent = parent;

        for (i, ((hash, depth), data)) in names.iter().zip(blocks).enumerate() {
            match self.admit(*hash, parent, *depth, data) {
                Admit::Inserted => report.inserted += 1,
                Admit::AlreadyPresent => report.deduped += 1,
                // Everything after a failure is an orphan.
                Admit::OutOfSpace | Admit::OrphanParent => {
                    report.dropped += blocks.len() - i;
                    break;
                }
            }
            parent = Some(*hash);
        }
        report
    }
}
