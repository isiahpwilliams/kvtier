//! Name a sequence, find the longest resident prefix, admit what was
//! missing. The whole cache minus the network, the tiers and eviction.

use std::io;

use crate::block::{BlockHash, BlockLayout, PrefixHasher, TokenId};
use crate::index::Index;
use crate::slab::Slab;

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
    /// Admits refused because the slab was full.
    pub rejected_blocks: u64,
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
    /// The slab is full. Phase 3 turns this into an eviction decision.
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
    layout: BlockLayout,
    hasher: PrefixHasher,
    slab: Slab,
    index: Index,
    stats: Stats,
}

impl KvStore {
    pub fn new(model_id: &str, layout: BlockLayout, capacity_blocks: usize) -> io::Result<Self> {
        let hasher = PrefixHasher::new(model_id, &layout);
        let slab = Slab::new(layout.block_bytes(), capacity_blocks)?;
        Ok(Self {
            layout,
            hasher,
            slab,
            index: Index::new(),
            stats: Stats::default(),
        })
    }

    pub fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    pub fn stats(&self) -> Stats {
        self.stats
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

        self.stats.lookups += 1;
        self.stats.queried_blocks += hashes.len() as u64;
        self.stats.hit_blocks += matched as u64;

        matched
    }

    /// Borrow a resident block's bytes. Phase 2 writes this slice straight
    /// to a socket.
    pub fn read(&self, hash: BlockHash) -> Option<&[u8]> {
        let slot = self.index.get(hash)?.slot;
        Some(self.slab.block(slot))
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
        let Some(slot) = self.slab.alloc() else {
            self.stats.rejected_blocks += 1;
            return Admit::OutOfSpace;
        };

        self.slab.block_mut(slot).copy_from_slice(data);
        self.index
            .insert(hash, parent, slot, depth_tokens)
            .expect("residency and duplication were checked above");

        self.stats.inserted_blocks += 1;
        self.stats.bytes_admitted += data.len() as u64;
        Admit::Inserted
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

        let mut report = AdmitReport::default();
        for (i, data) in blocks.iter().enumerate() {
            let parent = if i == 0 { None } else { Some(hashes[i - 1]) };
            let depth = ((i + 1) * self.layout.tokens_per_block) as u32;

            match self.admit(hashes[i], parent, depth, data) {
                Admit::Inserted => report.inserted += 1,
                Admit::AlreadyPresent => report.deduped += 1,
                // Everything after a failure is an orphan.
                Admit::OutOfSpace | Admit::OrphanParent => {
                    report.dropped += blocks.len() - i;
                    break;
                }
            }
        }
        report
    }
}
