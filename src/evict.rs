//! Eviction policy: which block to give up when the slab is full.
//!
//! GreedyDual-Size (Cao & Irani, 1997) with uniform size, which puts recency
//! and recompute cost in one number:
//!
//!   on access:  H(b) = L + cost(b)
//!   on evict:   pick min H, then set L = H(victim)
//!
//! `L` is a monotone waterline. Touching a block lifts it clear; an untouched
//! one keeps its score while the water rises around it. Cost decides how high
//! above the line a block starts.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::block::BlockHash;
use crate::index::{Entry, Index};

/// What it costs to recompute one block, given its prefix is still cached,
/// in units where a block at depth zero costs 1.
///
/// The linear layers cost the same at any depth; attention is proportional to
/// it. They are equal at `attention_crossover_tokens` -- about 30k for a
/// Llama-3-8B shape -- so the spread across realistic depths is only 1x to 5x,
/// and cost modifies recency rather than dominating it.
#[derive(Clone, Copy, Debug)]
pub struct CostModel {
    pub attention_crossover_tokens: f64,
}

impl CostModel {
    /// Sensible for 7B-13B models. Longer-context or thinner models shift it.
    pub const DEFAULT_CROSSOVER: f64 = 30_000.0;

    pub fn recompute_cost(&self, depth_tokens: u32) -> f64 {
        1.0 + f64::from(depth_tokens) / self.attention_crossover_tokens
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            attention_crossover_tokens: Self::DEFAULT_CROSSOVER,
        }
    }
}

/// Heap entry. Positive finite floats order the same as their IEEE bit
/// patterns, so sorting on `u64` gives `Ord` without a wrapper type.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    key: u64,
    hash: BlockHash,
}

fn key(priority: f64) -> u64 {
    debug_assert!(
        priority.is_finite() && priority >= 0.0,
        "priority {priority}"
    );
    priority.to_bits()
}

pub struct GreedyDual {
    cost: CostModel,
    /// `L`: the priority of the last block evicted.
    inflation: f64,
    /// Candidate leaves, min-first. Entries go stale when a block is touched,
    /// gains a child, or disappears; `select_victim` filters them at pop time
    /// rather than trying to keep the heap exact.
    heap: BinaryHeap<Reverse<Candidate>>,
}

impl GreedyDual {
    pub fn new(cost: CostModel) -> Self {
        Self {
            cost,
            inflation: 0.0,
            heap: BinaryHeap::new(),
        }
    }

    pub fn inflation(&self) -> f64 {
        self.inflation
    }

    /// What a block at this depth is worth right now.
    pub fn priority_for(&self, depth_tokens: u32) -> f64 {
        self.inflation + self.cost.recompute_cost(depth_tokens)
    }

    pub fn offer(&mut self, hash: BlockHash, priority: f64) {
        self.heap.push(Reverse(Candidate {
            key: key(priority),
            hash,
        }));
    }

    /// The cheapest block worth giving up, or `None` if nothing is eligible.
    ///
    /// `protect` is the parent the caller is about to attach a child to: a
    /// leaf until that child lands, so otherwise the likeliest victim, and
    /// taking it would orphan the incoming block.
    pub fn select_victim(
        &mut self,
        index: &Index,
        protect: Option<BlockHash>,
        accept: impl Fn(&Entry) -> bool,
    ) -> Option<(BlockHash, f64)> {
        let mut deferred = Vec::new();
        let mut victim = None;

        while let Some(Reverse(candidate)) = self.heap.pop() {
            let Some(entry) = index.get(candidate.hash) else {
                continue; // already gone
            };
            if key(entry.priority) != candidate.key {
                continue; // touched since; a fresher candidate is in the heap
            }
            // A pin is temporary and carries no reprice when it lifts, so
            // dropping these would quietly drain the heap of every block
            // writeback ever touched.
            if entry.pins > 0 || Some(candidate.hash) == protect {
                deferred.push(Reverse(candidate));
                continue;
            }
            if !accept(entry) {
                continue; // moved tier, and repriced onto the other heap
            }
            victim = Some((candidate.hash, entry.priority));
            break;
        }

        self.heap.extend(deferred);
        victim
    }

    /// The `count` lowest-priority blocks matching `accept`, left in place.
    ///
    /// Writeback uses this to clean the blocks nearest the eviction frontier,
    /// so the ones most likely to go are already safe on disk by the time
    /// their turn comes.
    pub fn peek_victims(
        &mut self,
        index: &Index,
        count: usize,
        accept: impl Fn(&Entry) -> bool,
    ) -> Vec<BlockHash> {
        let mut found = Vec::with_capacity(count);
        let mut seen = Vec::new();

        while found.len() < count {
            let Some(Reverse(candidate)) = self.heap.pop() else {
                break;
            };
            let Some(entry) = index.get(candidate.hash) else {
                continue; // gone
            };
            if key(entry.priority) != candidate.key {
                continue; // stale
            }
            if entry.pins == 0 && accept(entry) {
                found.push(candidate.hash);
            }
            seen.push(Reverse(candidate));
        }

        // Nothing is evicted here, so everything still valid goes back.
        self.heap.extend(seen);
        found
    }

    pub fn note_eviction(&mut self, priority: f64) {
        debug_assert!(
            priority >= self.inflation,
            "inflation must not go backwards"
        );
        self.inflation = priority;
    }

    pub fn candidates(&self) -> usize {
        self.heap.len()
    }

    /// Every access pushes a fresh candidate and strands the old one, so a
    /// hot block leaves a trail. Rebuild from the index once the stale
    /// entries outnumber the real ones.
    pub fn should_compact(&self, resident_blocks: usize) -> bool {
        self.heap.len() > 64 && self.heap.len() > 4 * resident_blocks.max(1)
    }

    pub fn compact(&mut self, index: &Index, accept: impl Fn(&Entry) -> bool) {
        self.heap.clear();
        for (hash, entry) in index.iter().filter(|(_, e)| e.pins == 0 && accept(e)) {
            self.heap.push(Reverse(Candidate {
                key: key(entry.priority),
                hash,
            }));
        }
    }
}

impl Default for GreedyDual {
    fn default() -> Self {
        Self::new(CostModel::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_raises_recompute_cost_but_not_wildly() {
        let cost = CostModel::default();
        assert!((cost.recompute_cost(0) - 1.0).abs() < 1e-9);

        // A 4k prefix is only ~13% more expensive than a shallow block: the
        // linear layers dominate until the context gets long.
        let shallow = cost.recompute_cost(4_096);
        assert!(shallow > 1.13 && shallow < 1.15, "got {shallow}");

        // At 120k, attention has taken over and it is ~5x.
        let deep = cost.recompute_cost(120_000);
        assert!(deep > 4.9 && deep < 5.1, "got {deep}");
    }

    #[test]
    fn float_priorities_order_as_integers() {
        let mut previous = 0u64;
        for step in 0..64 {
            let current = key(1.0 + f64::from(step) / 7.0);
            assert!(current > previous, "bit keys must be monotone");
            previous = current;
        }
    }
}
