//! Block name -> location. Prefix-chained names give a radix tree over token
//! sequences for free: probe block 0, block 1, ... and stop at the first
//! miss; the depth reached is the hit length.
//!
//! The parent link and child count exist for one invariant ordinary caches
//! lack: a block is useless without every block before it, since attention
//! over token 5000 reads the KV of tokens 0..5000. So we insert only when the
//! parent is resident, and remove only leaves.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::block::BlockHash;
use crate::slab::SlotId;

/// Passes a `BlockHash`'s bucket key straight through. The keys are already
/// BLAKE3 digests, so `HashMap`'s default SipHash pass would add no
/// randomness on the hottest path in the system.
#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }

    /// Only reached with other key types; degrades rather than panicking.
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }
}

type BlockMap<V> = HashMap<BlockHash, V, BuildHasherDefault<IdentityHasher>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub slot: SlotId,
    /// `None` for a sequence's first block: its parent is the unstored root.
    pub parent: Option<BlockHash>,
    /// Resident blocks naming this one as parent; only a leaf may be evicted.
    pub children: u32,
    /// Tokens covered from the start of the sequence through this block.
    /// Phase 3 prices recompute cost with it: prefill is superlinear in depth.
    pub depth_tokens: u32,
    /// Logical clock value at last hit.
    pub last_access: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum IndexError {
    AlreadyPresent,
    /// The named parent is not resident.
    OrphanParent,
    /// Removing this block would strand its descendants.
    HasChildren,
    NotFound,
}

pub struct Index {
    entries: BlockMap<Entry>,
    clock: u64,
}

impl Index {
    pub fn new() -> Self {
        Self {
            entries: BlockMap::default(),
            clock: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, hash: BlockHash) -> bool {
        self.entries.contains_key(&hash)
    }

    pub fn get(&self, hash: BlockHash) -> Option<&Entry> {
        self.entries.get(&hash)
    }

    /// Look up and record the access.
    pub fn touch(&mut self, hash: BlockHash) -> Option<&Entry> {
        self.clock += 1;
        let clock = self.clock;
        let entry = self.entries.get_mut(&hash)?;
        entry.last_access = clock;
        Some(entry)
    }

    pub fn insert(
        &mut self,
        hash: BlockHash,
        parent: Option<BlockHash>,
        slot: SlotId,
        depth_tokens: u32,
    ) -> Result<(), IndexError> {
        if self.entries.contains_key(&hash) {
            return Err(IndexError::AlreadyPresent);
        }
        if let Some(parent) = parent
            && !self.entries.contains_key(&parent)
        {
            return Err(IndexError::OrphanParent);
        }

        self.clock += 1;
        self.entries.insert(
            hash,
            Entry {
                slot,
                parent,
                children: 0,
                depth_tokens,
                last_access: self.clock,
            },
        );

        if let Some(parent) = parent {
            // Cannot fail: checked resident above.
            self.entries.get_mut(&parent).unwrap().children += 1;
        }
        Ok(())
    }

    /// Remove a leaf, returning its entry so the caller can free the slot.
    pub fn remove(&mut self, hash: BlockHash) -> Result<Entry, IndexError> {
        match self.entries.get(&hash) {
            None => return Err(IndexError::NotFound),
            Some(entry) if entry.children > 0 => return Err(IndexError::HasChildren),
            Some(_) => {}
        }

        let entry = self.entries.remove(&hash).unwrap();
        if let Some(parent) = entry.parent
            && let Some(parent_entry) = self.entries.get_mut(&parent)
        {
            parent_entry.children -= 1;
        }
        Ok(entry)
    }

    /// How many leading blocks of `hashes` are resident. Stops at the first
    /// miss: a hit past a gap is not a hit.
    pub fn match_prefix(&mut self, hashes: &[BlockHash]) -> usize {
        let mut matched = 0;
        for &hash in hashes {
            if self.touch(hash).is_none() {
                break;
            }
            matched += 1;
        }
        matched
    }

    /// Resident leaves: the only blocks eligible for eviction.
    pub fn leaves(&self) -> impl Iterator<Item = (BlockHash, &Entry)> {
        self.entries
            .iter()
            .filter(|(_, e)| e.children == 0)
            .map(|(h, e)| (*h, e))
    }
}

impl Default for Index {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockLayout, PrefixHasher, TokenId};

    fn slot(n: u32) -> SlotId {
        SlotId::for_tests(n)
    }

    fn chain(n: usize) -> Vec<BlockHash> {
        let layout = BlockLayout::tiny();
        let hasher = PrefixHasher::new("m", &layout);
        let tokens: Vec<TokenId> = (0..(n * 16) as TokenId).collect();
        hasher.chain(&tokens)
    }

    #[test]
    fn insert_requires_resident_parent() {
        let mut index = Index::new();
        let hashes = chain(3);

        assert_eq!(
            index.insert(hashes[1], Some(hashes[0]), slot(0), 32),
            Err(IndexError::OrphanParent),
            "must refuse a block whose prefix is missing"
        );

        index.insert(hashes[0], None, slot(0), 16).unwrap();
        index
            .insert(hashes[1], Some(hashes[0]), slot(1), 32)
            .unwrap();
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn duplicate_insert_is_rejected() {
        let mut index = Index::new();
        let hashes = chain(1);
        index.insert(hashes[0], None, slot(0), 16).unwrap();
        assert_eq!(
            index.insert(hashes[0], None, slot(1), 16),
            Err(IndexError::AlreadyPresent)
        );
    }

    #[test]
    fn parents_are_pinned_by_their_children() {
        let mut index = Index::new();
        let hashes = chain(3);
        index.insert(hashes[0], None, slot(0), 16).unwrap();
        index
            .insert(hashes[1], Some(hashes[0]), slot(1), 32)
            .unwrap();
        index
            .insert(hashes[2], Some(hashes[1]), slot(2), 48)
            .unwrap();

        assert_eq!(index.remove(hashes[0]), Err(IndexError::HasChildren));
        assert_eq!(index.remove(hashes[1]), Err(IndexError::HasChildren));

        // Peeling inward from the leaf works.
        index.remove(hashes[2]).unwrap();
        index.remove(hashes[1]).unwrap();
        index.remove(hashes[0]).unwrap();
        assert!(index.is_empty());
    }

    #[test]
    fn only_leaves_are_evictable() {
        let mut index = Index::new();
        let hashes = chain(3);
        index.insert(hashes[0], None, slot(0), 16).unwrap();
        index
            .insert(hashes[1], Some(hashes[0]), slot(1), 32)
            .unwrap();
        index
            .insert(hashes[2], Some(hashes[1]), slot(2), 48)
            .unwrap();

        let leaves: Vec<_> = index.leaves().map(|(h, _)| h).collect();
        assert_eq!(leaves, vec![hashes[2]]);
    }

    #[test]
    fn match_prefix_stops_at_first_gap() {
        let mut index = Index::new();
        let hashes = chain(4);
        index.insert(hashes[0], None, slot(0), 16).unwrap();
        index
            .insert(hashes[1], Some(hashes[0]), slot(1), 32)
            .unwrap();

        assert_eq!(index.match_prefix(&hashes), 2);
        assert_eq!(index.match_prefix(&hashes[..1]), 1);
        assert_eq!(index.match_prefix(&[]), 0);
    }

    #[test]
    fn touch_advances_recency() {
        let mut index = Index::new();
        let hashes = chain(2);
        index.insert(hashes[0], None, slot(0), 16).unwrap();
        index
            .insert(hashes[1], Some(hashes[0]), slot(1), 32)
            .unwrap();

        let before = index.get(hashes[0]).unwrap().last_access;
        index.touch(hashes[0]).unwrap();
        assert!(index.get(hashes[0]).unwrap().last_access > before);
    }
}
