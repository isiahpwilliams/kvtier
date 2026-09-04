//! Name a sequence, find the longest resident prefix, admit what was
//! missing. The whole cache minus the network, the tiers and eviction.

use std::io;

use crate::block::{BlockHash, BlockLayout, PrefixHasher, TokenId};
use crate::evict::{CostModel, GreedyDual};
use crate::index::{Index, Place};
use crate::slab::{PinnedBlock, Slab};
use crate::tier::{DiskStats, DiskTier, TierCosts};

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
        policy.offer(hash, priority);
        let crowded = policy.should_compact(self.index.len());
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
            .insert(hash, parent, Place::Ram(slot), depth_tokens)
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

            // Demotion beats dropping whenever a read back is cheaper than a
            // rebuild, and unlike dropping it works on any block: a demoted
            // parent is still there for its children.
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
        self.reprice(hash);
        Ok(())
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
        disk.free(disk_slot);

        self.index.set_place(hash, Place::Ram(slot));
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
        match entry.place {
            Place::Ram(slot) => self.slab.free(slot),
            Place::Disk(slot) => self
                .disk
                .as_mut()
                .expect("a disk block needs a tier")
                .free(slot),
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
