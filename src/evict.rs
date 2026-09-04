//! Eviction policy: which block to give up when the slab is full.
//!
//! LRU is the wrong default here because blocks are not equally expensive to
//! replace. The policy is GreedyDual-Size (Cao & Irani, 1997) with uniform
//! size, which combines recency and cost in one number:
//!
//!   on access:  H(b) = L + cost(b)
//!   on evict:   pick min H, then set L = H(victim)
//!
//! `L` is a monotone inflation clock. A block touched recently carries a high
//! `H`; one that has sat since `L` was low carries a stale low `H` and goes
//! first. Cost tilts that ordering toward keeping what is expensive to
//! rebuild.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::block::BlockHash;
use crate::index::Index;

/// What it costs to recompute one block, given its prefix is still cached.
///
/// Two terms, in units where a block at depth zero costs 1:
///
///   * the linear layers, which cost the same at any depth
///   * attention, which is proportional to how many tokens precede the block
///
/// They are equal at `attention_crossover_tokens`. For a Llama-3-8B-shaped
/// model that is around 30k tokens: 2*P FLOPs per token for the linear terms
/// against 4*D*d_model*layers for attention. So the spread across realistic
/// depths is roughly 1x to 5x -- real, but nothing like the 400x you get if
/// you assume attention dominates everywhere.
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

/// Heap entry. Priorities are positive and finite, and for those the IEEE bit
/// pattern orders exactly like the float, so we can sort on `u64` and keep
/// `Ord` without pulling in a wrapper type.
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

    /// Offer a block as an eviction candidate at the given priority.
    pub fn offer(&mut self, hash: BlockHash, priority: f64) {
        self.heap.push(Reverse(Candidate {
            key: key(priority),
            hash,
        }));
    }

    /// The cheapest block worth giving up, or `None` if nothing is eligible.
    ///
    /// `protect` is the block the caller is about to attach a child to.
    /// Without it the parent -- a leaf until its child lands -- is usually the
    /// most attractive victim, and admitting would orphan its own block.
    pub fn select_victim(
        &mut self,
        index: &Index,
        protect: Option<BlockHash>,
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
            if entry.children > 0 || entry.pins > 0 {
                continue; // re-offered when it becomes a leaf again
            }
            if Some(candidate.hash) == protect {
                deferred.push(Reverse(candidate));
                continue;
            }
            victim = Some((candidate.hash, entry.priority));
            break;
        }

        self.heap.extend(deferred);
        victim
    }

    /// Record that `priority` was the cost of the block we gave up.
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

    pub fn compact(&mut self, index: &Index) {
        self.heap.clear();
        for (hash, entry) in index.leaves() {
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
