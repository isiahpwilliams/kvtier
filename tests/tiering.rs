//! Demotion to disk, promotion back, and the choice between them and dropping.

use kvtier::block::{BlockHash, BlockLayout, PrefixHasher, TokenId};
use kvtier::store::{Admit, KvStore};
use kvtier::tier::{DiskTier, TierCosts};
use kvtier::trace::{self, SplitMix64};

const MODEL: &str = "tier-test";

fn hasher() -> PrefixHasher {
    PrefixHasher::new(MODEL, &BlockLayout::tiny())
}

fn tokens(count: usize, seed: u64) -> Vec<TokenId> {
    SplitMix64::new(seed).tokens(count, 32_000)
}

fn tiered(ram_blocks: usize, disk_blocks: usize) -> KvStore {
    let layout = BlockLayout::tiny();
    let disk = DiskTier::temporary(layout.block_bytes(), disk_blocks).unwrap();
    KvStore::new(MODEL, layout, ram_blocks)
        .unwrap()
        .with_disk_tier(disk)
}

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

fn expected(store: &KvStore, hash: BlockHash) -> Vec<u8> {
    trace::block_payload(hash, store.layout().block_bytes())
}

/// Blocks the store still holds but has pushed out of RAM. Which ones those
/// are is the policy's business, not the test's.
fn demoted(store: &KvStore, hashes: &[BlockHash]) -> Vec<BlockHash> {
    hashes
        .iter()
        .copied()
        .filter(|&hash| store.contains(hash) && store.read(hash).is_none())
        .collect()
}

/// Fill a store past its RAM capacity and report every block admitted.
fn overfill(store: &mut KvStore, count: u64) -> Vec<BlockHash> {
    overfill_from(store, 1, count)
}

/// Same, from a seed range that will not collide with an earlier call.
fn overfill_from(store: &mut KvStore, start: u64, count: u64) -> Vec<BlockHash> {
    (start..start + count)
        .map(|seed| admit_leaf(store, seed, 16))
        .collect()
}

#[test]
fn blocks_pushed_out_of_ram_land_on_disk() {
    let mut store = tiered(4, 32);
    let admitted = overfill(&mut store, 7);

    assert_eq!(store.resident_blocks(), 7, "nothing was actually lost");
    assert_eq!(demoted(&store, &admitted).len(), 3);
    assert_eq!(store.disk_blocks(), 3);
    assert_eq!(store.stats().evicted_blocks, 0, "demoted, not dropped");
}

#[test]
fn a_demoted_block_still_counts_as_a_hit() {
    let mut store = tiered(4, 32);
    let admitted = overfill(&mut store, 7);
    let off_ram = demoted(&store, &admitted);

    assert_eq!(
        store.match_prefix(&off_ram),
        off_ram.len(),
        "a hit is a hit"
    );
}

#[test]
fn a_hit_on_a_demoted_block_reads_it_back() {
    let mut store = tiered(4, 32);
    let admitted = overfill(&mut store, 7);
    let target = demoted(&store, &admitted)[0];

    let pinned = store.pin_run(&[target]);
    assert_eq!(pinned.len(), 1);
    assert_eq!(
        pinned[0].bytes(),
        expected(&store, target).as_slice(),
        "bytes must survive the round trip through the file"
    );
    store.unpin_all(&pinned);

    assert_eq!(store.stats().promoted_blocks, 1);
    assert!(store.read(target).is_some(), "back in RAM");
    assert_eq!(store.disk_stats().unwrap().reads, 1);
}

#[test]
fn a_block_too_cheap_to_be_worth_a_disk_slot_is_dropped() {
    // With rebuilding nearly free, a read back costs more than a recompute,
    // so the disk slot does not pay for itself.
    let layout = BlockLayout::tiny();
    let disk = DiskTier::temporary(layout.block_bytes(), 32).unwrap();
    let mut store = KvStore::new(MODEL, layout, 2)
        .unwrap()
        .with_disk_tier(disk)
        .with_tier_costs(TierCosts {
            recompute_secs: 1e-9,
            ..TierCosts::default()
        });

    let admitted = overfill(&mut store, 3);

    assert_eq!(store.resident_blocks(), 2, "one block is simply gone");
    assert_eq!(demoted(&store, &admitted).len(), 0);
    assert_eq!(store.disk_blocks(), 0, "nothing was written to disk");
    assert_eq!(store.stats().demoted_blocks, 0);
    assert_eq!(store.stats().evicted_blocks, 1);
}

#[test]
fn demotion_can_take_a_block_that_dropping_could_not() {
    // Dropping is leaf-only: an internal block would strand its children.
    // Demotion has no such constraint, because a demoted parent is still
    // there for them.
    let mut store = tiered(4, 32);
    let block_bytes = store.layout().block_bytes();

    // One chain of 4 blocks, so blocks 0..2 are internal.
    let sequence = tokens(64, 10);
    let hashes = hasher().chain(&sequence);
    let payloads: Vec<Vec<u8>> = hashes
        .iter()
        .map(|&hash| trace::block_payload(hash, block_bytes))
        .collect();
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    store.admit_sequence(&sequence, &refs);
    assert_eq!(store.resident_blocks(), 4);

    // A block from an unrelated chain forces something out of RAM.
    admit_leaf(&mut store, 99, 16);

    assert_eq!(store.stats().evicted_blocks, 0, "nothing was lost");
    assert!(store.stats().demoted_blocks >= 1);
    assert_eq!(store.match_prefix(&hashes), 4, "the chain is still whole");
}

#[test]
fn a_full_disk_drops_its_cheapest_leaf() {
    let mut store = tiered(2, 2);
    let admitted = overfill(&mut store, 8);

    assert_eq!(store.resident_blocks(), 4, "two tiers of two");
    assert!(store.stats().evicted_blocks > 0, "the disk had to give too");
    assert!(
        store.contains(admitted[7]),
        "the newest block must be the one that survived"
    );
}

#[test]
fn a_promoted_block_can_be_demoted_again() {
    let mut store = tiered(2, 8);
    let kept = admit_leaf(&mut store, 1, 16);
    let mut seed = 100;

    for round in 0..3 {
        // Let it go cold first: one admit per round would leave it the most
        // recently used block in RAM, and it would never leave.
        for _ in 0..3 {
            admit_leaf(&mut store, seed, 16);
            seed += 1;
        }
        assert!(
            store.read(kept).is_none(),
            "should be on disk by round {round}"
        );

        // The pin pulls it back, and the next round pushes it out again.
        let pinned = store.pin_run(&[kept]);
        assert_eq!(pinned.len(), 1, "still reachable on round {round}");
        assert_eq!(pinned[0].bytes(), expected(&store, kept).as_slice());
        store.unpin_all(&pinned);
    }

    let stats = store.stats();
    assert!(stats.promoted_blocks >= 2 && stats.demoted_blocks >= 2);
    assert!(store.contains(kept));
}

#[test]
fn a_tiered_store_holds_far_more_than_its_ram() {
    let mut store = tiered(8, 256);
    overfill(&mut store, 200);

    assert_eq!(store.resident_blocks(), 200, "all of it is still cached");
    assert_eq!(store.disk_blocks(), 192);
    assert_eq!(store.stats().evicted_blocks, 0, "nothing needed dropping");
}

#[test]
fn a_fetch_reserves_then_publishes() {
    let mut store = tiered(4, 32);
    let admitted = overfill(&mut store, 7);
    let off_ram = demoted(&store, &admitted);
    assert!(!off_ram.is_empty());

    // Phase one: no I/O has happened, so the block is still on disk.
    let mut parts = store.begin_fetch(&off_ram);
    assert_eq!(parts.len(), off_ram.len());
    assert_eq!(store.disk_stats().unwrap().reads, 0);

    // Phase two, which the server runs with the lock released.
    let reader = store.disk_reader().unwrap();
    for part in &mut parts {
        if let kvtier::store::FetchPart::Pending(promotion) = part {
            promotion.fill(&reader).unwrap();
        }
    }

    // Phase three: publish and hand back the run.
    let pinned = store.finish_fetch(parts);
    assert_eq!(pinned.len(), off_ram.len());
    for block in &pinned {
        assert_eq!(block.bytes(), expected(&store, block.hash()).as_slice());
    }
    store.unpin_all(&pinned);
    assert_eq!(store.stats().promoted_blocks as usize, off_ram.len());
}

#[test]
fn an_unfilled_block_truncates_the_run_instead_of_serving_a_hole() {
    let mut store = tiered(4, 32);
    let admitted = overfill(&mut store, 7);
    let off_ram = demoted(&store, &admitted);
    assert!(off_ram.len() >= 2);

    // Fill everything except the first demoted block, as a failed read would.
    let reader = store.disk_reader().unwrap();
    let mut parts = store.begin_fetch(&off_ram);
    let mut seen_pending = false;
    for part in &mut parts {
        if let kvtier::store::FetchPart::Pending(promotion) = part {
            if seen_pending {
                promotion.fill(&reader).unwrap();
            }
            seen_pending = true;
        }
    }

    let pinned = store.finish_fetch(parts);
    assert!(
        pinned.len() < off_ram.len(),
        "the run must stop at the hole"
    );
    store.unpin_all(&pinned);

    // Everything reserved was given back: no leaked pins, no leaked slots.
    assert_eq!(store.resident_blocks(), 7);
    for &hash in &admitted {
        assert!(store.contains(hash));
    }
}

#[test]
fn a_block_being_read_cannot_be_evicted_underneath() {
    let mut store = tiered(4, 8);
    let admitted = overfill(&mut store, 6);
    let off_ram = demoted(&store, &admitted);

    // Reserve the run, then pile on pressure before the reads land.
    let mut parts = store.begin_fetch(&off_ram);
    for seed in 100..120 {
        admit_leaf(&mut store, seed, 16);
    }

    let reader = store.disk_reader().unwrap();
    for part in &mut parts {
        if let kvtier::store::FetchPart::Pending(promotion) = part {
            promotion.fill(&reader).unwrap();
        }
    }

    let pinned = store.finish_fetch(parts);
    assert_eq!(
        pinned.len(),
        off_ram.len(),
        "pins held through the pressure"
    );
    for block in &pinned {
        assert_eq!(block.bytes(), expected(&store, block.hash()).as_slice());
    }
    store.unpin_all(&pinned);
}

/// Drive one writeback round the way the server's loop does.
fn run_writeback(store: &mut KvStore, max: usize) -> usize {
    let writer = store.disk_writer().unwrap();
    let mut jobs = store.begin_writeback(max);
    let count = jobs.len();
    for job in &mut jobs {
        job.flush(&writer).unwrap();
    }
    store.finish_writeback(jobs);
    count
}

#[test]
fn a_freshly_admitted_block_is_dirty_until_written_back() {
    let mut store = tiered(4, 32);
    overfill(&mut store, 4);
    assert_eq!(store.dirty_blocks(), 4, "nothing has been copied yet");

    assert_eq!(run_writeback(&mut store, 4), 4);
    assert_eq!(store.dirty_blocks(), 0);
    assert_eq!(store.stats().written_back, 4);
    assert_eq!(store.resident_blocks(), 4, "still all in RAM");
}

#[test]
fn evicting_a_clean_block_costs_no_write() {
    let mut store = tiered(4, 32);
    overfill(&mut store, 4);
    run_writeback(&mut store, 4);

    // RAM is full and every block already has a copy on disk.
    admit_leaf(&mut store, 99, 16);

    let stats = store.stats();
    assert_eq!(stats.demoted_blocks, 1, "something left RAM");
    assert_eq!(
        stats.blocking_demotions, 0,
        "and it did not write under the lock"
    );
}

#[test]
fn without_writeback_the_admit_path_pays_for_the_write() {
    // The behaviour writeback exists to remove, kept as the contrast.
    let mut store = tiered(4, 32);
    overfill(&mut store, 5);

    let stats = store.stats();
    assert_eq!(stats.demoted_blocks, 1);
    assert_eq!(stats.blocking_demotions, 1);
    assert_eq!(stats.written_back, 0);
}

#[test]
fn writeback_cleans_only_what_it_is_asked_for() {
    let mut store = tiered(8, 32);
    overfill(&mut store, 8);

    assert_eq!(run_writeback(&mut store, 3), 3);
    assert_eq!(store.dirty_blocks(), 5);
    assert_eq!(
        run_writeback(&mut store, 100),
        5,
        "the rest on the next pass"
    );
    assert_eq!(store.dirty_blocks(), 0);
}

#[test]
fn an_unflushed_job_leaves_its_block_dirty_and_frees_the_slot() {
    let mut store = tiered(4, 32);
    overfill(&mut store, 4);

    // Reserve the disk slots, then fail every write.
    let jobs = store.begin_writeback(4);
    assert_eq!(jobs.len(), 4);
    store.finish_writeback(jobs);

    assert_eq!(store.dirty_blocks(), 4, "still needs writing");
    assert_eq!(store.stats().written_back, 0);
    assert_eq!(store.disk_blocks(), 0, "reserved slots were given back");
}

#[test]
fn a_written_back_block_reads_correctly_after_eviction() {
    let mut store = tiered(4, 32);
    let admitted = overfill(&mut store, 4);
    run_writeback(&mut store, 4);

    // Push them all out of RAM, using only the disk copies made above.
    overfill_from(&mut store, 500, 8);

    let off_ram = demoted(&store, &admitted);
    assert!(!off_ram.is_empty());
    let pinned = store.pin_run(&off_ram);
    assert_eq!(pinned.len(), off_ram.len());
    for block in &pinned {
        assert_eq!(
            block.bytes(),
            expected(&store, block.hash()).as_slice(),
            "the copy writeback made must be the real bytes"
        );
    }
    store.unpin_all(&pinned);
}

#[test]
fn a_promoted_block_keeps_its_disk_copy() {
    // It came off disk, so the copy is already correct. Keeping it means
    // pushing the block back out is free.
    let mut store = tiered(4, 32);
    let admitted = overfill(&mut store, 7);
    let target = demoted(&store, &admitted)[0];

    let pinned = store.pin_run(&[target]);
    store.unpin_all(&pinned);

    assert!(
        !store.index().get(target).unwrap().place.is_dirty(),
        "a block read off disk is already clean, so pushing it back out is free"
    );
}
