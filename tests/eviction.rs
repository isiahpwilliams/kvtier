//! Eviction behaviour: what the store gives up when it runs out of room.

use kvtier::block::{BlockHash, BlockLayout, PrefixHasher, TokenId};
use kvtier::store::{Admit, KvStore};
use kvtier::trace::{self, SplitMix64};

const MODEL: &str = "evict-test";

fn new_store(capacity_blocks: usize) -> KvStore {
    KvStore::new(MODEL, BlockLayout::tiny(), capacity_blocks).unwrap()
}

fn hasher() -> PrefixHasher {
    PrefixHasher::new(MODEL, &BlockLayout::tiny())
}

fn tokens(count: usize, seed: u64) -> Vec<TokenId> {
    SplitMix64::new(seed).tokens(count, 32_000)
}

/// Admit every full block of a sequence and report what the store did.
fn admit_all(store: &mut KvStore, tokens: &[TokenId]) -> kvtier::store::AdmitReport {
    let block_bytes = store.layout().block_bytes();
    let hashes = hasher().chain(tokens);
    let payloads: Vec<Vec<u8>> = hashes
        .iter()
        .map(|&hash| trace::block_payload(hash, block_bytes))
        .collect();
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    store.admit_sequence(tokens, &refs)
}

/// A one-block chain with no parent, so every one of these is a leaf.
fn admit_leaf(store: &mut KvStore, seed: u64, depth_tokens: u32) -> BlockHash {
    let block_bytes = store.layout().block_bytes();
    let hash = hasher().chain(&tokens(16, seed))[0];
    let payload = trace::block_payload(hash, block_bytes);
    assert_eq!(
        store.admit(hash, None, depth_tokens, &payload),
        Admit::Inserted
    );
    hash
}

#[test]
fn a_full_store_makes_room_instead_of_refusing() {
    let mut store = new_store(8);
    for seed in 0..64 {
        admit_leaf(&mut store, seed, 16);
    }

    assert_eq!(store.resident_blocks(), 8, "must stay at capacity");
    assert!(store.stats().evicted_blocks >= 56);
    assert_eq!(
        store.stats().rejected_blocks,
        0,
        "there was always a victim"
    );
}

#[test]
fn the_cheapest_block_to_rebuild_goes_first() {
    let mut store = new_store(2);
    // Same recency, wildly different depth: one sits at the start of a
    // context, the other 120k tokens in.
    let shallow = admit_leaf(&mut store, 1, 0);
    let deep = admit_leaf(&mut store, 2, 120_000);

    admit_leaf(&mut store, 3, 16);

    assert!(store.read(shallow).is_none(), "the cheap block should go");
    assert!(store.read(deep).is_some(), "the expensive one should stay");
}

#[test]
fn a_block_that_keeps_getting_used_is_never_evicted() {
    let mut store = new_store(8);
    let kept = admit_leaf(&mut store, 999, 16);

    for seed in 0..200 {
        admit_leaf(&mut store, seed, 16);
        // One hit per round is enough to keep it ahead of the inflation clock.
        assert_eq!(store.match_prefix(&[kept]), 1, "evicted on round {seed}");
    }
    assert!(store.read(kept).is_some());
}

#[test]
fn a_pinned_block_survives_pressure() {
    let mut store = new_store(4);
    let pinned_hash = admit_leaf(&mut store, 1, 16);
    let pinned = store.pin_run(&[pinned_hash]);
    assert_eq!(pinned.len(), 1);

    for seed in 10..40 {
        admit_leaf(&mut store, seed, 16);
    }

    // The bytes a reader is holding must still be the right ones.
    let expected = trace::block_payload(pinned_hash, store.layout().block_bytes());
    assert_eq!(pinned[0].bytes(), expected.as_slice());

    store.unpin_all(&pinned);
    assert!(store.read(pinned_hash).is_some());
}

#[test]
fn a_blocks_own_parent_is_never_the_victim() {
    // The parent is a leaf right up until its child lands, which makes it the
    // most attractive victim in the store. Evicting it would orphan the very
    // block being admitted.
    let mut store = new_store(4);
    for seed in 0..16 {
        let report = admit_all(&mut store, &tokens(64, seed));
        assert_eq!(
            report.inserted + report.deduped + report.dropped,
            4,
            "every block accounted for"
        );
    }

    // Whatever survived must still be a connected chain, never an orphan.
    let index = store.index();
    for (hash, _) in index.leaves() {
        let mut current = index.get(hash).unwrap();
        while let Some(parent) = current.parent {
            current = index
                .get(parent)
                .unwrap_or_else(|| panic!("{hash:?} is orphaned"));
        }
    }
}

#[test]
fn a_cheap_newcomer_can_still_displace_an_expensive_resident() {
    // Refusing on cost alone would ossify the cache: every new block starts
    // shallow, so it would always lose to the deep tails already resident,
    // nothing would be admitted, nothing evicted, and the inflation clock
    // that breaks the tie would never rise. So the newcomer gets in.
    let mut store = new_store(2);
    admit_leaf(&mut store, 1, 120_000);
    admit_leaf(&mut store, 2, 120_000);

    let block_bytes = store.layout().block_bytes();
    let hash = hasher().chain(&tokens(16, 3))[0];
    let payload = trace::block_payload(hash, block_bytes);

    assert_eq!(store.admit(hash, None, 0, &payload), Admit::Inserted);
    assert_eq!(store.resident_blocks(), 2);
    assert_eq!(store.stats().evicted_blocks, 1);
}

#[test]
fn a_pinned_full_store_refuses_rather_than_evicting() {
    // Nothing is eligible, so there is genuinely no room. The store must say
    // so rather than break a pin.
    let mut store = new_store(2);
    let first = admit_leaf(&mut store, 1, 16);
    let second = admit_leaf(&mut store, 2, 16);
    let pinned = store.pin_run(&[first]);
    let also_pinned = store.pin_run(&[second]);

    let block_bytes = store.layout().block_bytes();
    let hash = hasher().chain(&tokens(16, 3))[0];
    let payload = trace::block_payload(hash, block_bytes);
    assert_eq!(store.admit(hash, None, 16, &payload), Admit::OutOfSpace);

    store.unpin_all(&pinned);
    store.unpin_all(&also_pinned);
}

#[test]
fn a_shared_system_prompt_outlives_the_conversations_using_it() {
    // The payoff test. One prompt, many short-lived conversations, a store
    // far too small to hold them all. The prompt must survive the churn.
    let mut store = new_store(24);

    let system_prompt = tokens(64, 42); // 4 blocks
    admit_all(&mut store, &system_prompt);

    for seed in 0..100 {
        let mut conversation = system_prompt.clone();
        conversation.extend(tokens(64, 1000 + seed));
        admit_all(&mut store, &conversation);
    }

    let prompt_hashes = hasher().chain(&system_prompt);
    assert_eq!(
        store.match_prefix(&prompt_hashes),
        4,
        "the shared prefix must still be resident after the churn"
    );
    assert!(
        store.stats().evicted_blocks > 100,
        "there was real pressure"
    );
}

#[test]
fn the_candidate_heap_does_not_grow_without_bound() {
    let mut store = new_store(16);
    let hashes: Vec<BlockHash> = (0..16)
        .map(|seed| admit_leaf(&mut store, seed, 16))
        .collect();

    // Hammer the same blocks: every hit offers a fresh candidate and strands
    // the previous one.
    for _ in 0..500 {
        store.match_prefix(&hashes);
    }
    assert!(
        store.eviction_candidates() <= 4 * store.resident_blocks() + 64,
        "heap grew to {} for {} blocks",
        store.eviction_candidates(),
        store.resident_blocks()
    );
}
