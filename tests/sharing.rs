//! End-to-end behaviour of the Phase 1 store.

use kvtier::block::{BlockLayout, TokenId};
use kvtier::store::{Admit, KvStore};
use kvtier::trace::{self, SplitMix64, WorkloadConfig};

const TPB: usize = 16;

/// Admit every full block of a sequence, generating payloads from names.
fn admit_all(store: &mut KvStore, tokens: &[TokenId]) {
    let block_bytes = store.layout().block_bytes();
    let lookup = store.lookup(tokens);
    let payloads: Vec<Vec<u8>> = lookup
        .hashes
        .iter()
        .map(|&hash| trace::block_payload(hash, block_bytes))
        .collect();
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    store.admit_sequence(tokens, &refs);
}

fn new_store(capacity_blocks: usize) -> KvStore {
    KvStore::new("test-model", BlockLayout::tiny(), capacity_blocks).unwrap()
}

#[test]
fn cold_lookup_misses_everything() {
    let mut store = new_store(64);
    let mut rng = SplitMix64::new(1);
    let tokens = rng.tokens(64, 32_000);

    let lookup = store.lookup(&tokens);
    assert_eq!(lookup.hashes.len(), 4);
    assert_eq!(lookup.matched, 0);
    assert_eq!(lookup.hit_tokens(), 0);
    assert_eq!(lookup.missing().len(), 4);
}

#[test]
fn round_trip_returns_the_bytes_that_went_in() {
    let mut store = new_store(64);
    let mut rng = SplitMix64::new(2);
    let tokens = rng.tokens(64, 32_000);
    admit_all(&mut store, &tokens);

    let lookup = store.lookup(&tokens);
    assert!(lookup.is_full_hit());
    for &hash in &lookup.hashes {
        let expected = trace::block_payload(hash, store.layout().block_bytes());
        assert_eq!(store.read(hash).unwrap(), expected.as_slice());
    }
}

#[test]
fn a_second_request_sharing_a_prefix_hits_on_it() {
    let mut store = new_store(64);
    let mut rng = SplitMix64::new(3);

    let system_prompt = rng.tokens(48, 32_000); // 3 blocks
    let mut first = system_prompt.clone();
    first.extend(rng.tokens(32, 32_000));
    let mut second = system_prompt.clone();
    second.extend(rng.tokens(32, 32_000));

    admit_all(&mut store, &first);

    let lookup = store.lookup(&second);
    assert_eq!(lookup.matched, 3, "the shared system prompt must hit");
    assert_eq!(lookup.hit_tokens(), 48);
    assert_eq!(lookup.missing().len(), 2, "and only the new turn must miss");
}

#[test]
fn shared_prefixes_are_stored_once() {
    let mut store = new_store(512);
    let mut rng = SplitMix64::new(4);

    let system_prompt = rng.tokens(64, 32_000); // 4 blocks
    let conversations = 10;

    for _ in 0..conversations {
        let mut tokens = system_prompt.clone();
        tokens.extend(rng.tokens(32, 32_000)); // 2 private blocks
        admit_all(&mut store, &tokens);
    }

    let naive = conversations * 6;
    assert_eq!(store.resident_blocks(), 4 + conversations * 2);
    assert_eq!(store.resident_blocks(), 24);
    assert!(store.resident_blocks() < naive);

    let stats = store.stats();
    assert_eq!(stats.deduped_blocks, ((conversations - 1) * 4) as u64);
}

#[test]
fn later_turns_hit_on_the_whole_conversation_so_far() {
    let mut store = new_store(4096);
    let config = WorkloadConfig {
        system_prompt_tokens: 128,
        conversations: 2,
        turns_per_conversation: 4,
        user_turn_tokens: 32,
        reply_tokens: 64,
        ..Default::default()
    };

    let mut hit_tokens_by_turn = vec![0usize; config.turns_per_conversation];
    for request in trace::generate(&config) {
        let lookup = store.lookup(&request.tokens);
        if request.conversation == 0 {
            hit_tokens_by_turn[request.turn] = lookup.hit_tokens();
        }
        // The engine hands back the reply's KV too, not just the prompt's.
        admit_all(&mut store, &request.completed);
    }

    // Turn 0 is cold; every later turn must hit more than the one before.
    assert_eq!(hit_tokens_by_turn[0], 0);
    for turn in 1..config.turns_per_conversation {
        assert!(
            hit_tokens_by_turn[turn] > hit_tokens_by_turn[turn - 1],
            "turn {turn} hit {} tokens, previous turn hit {}",
            hit_tokens_by_turn[turn],
            hit_tokens_by_turn[turn - 1]
        );
    }

    // By the last turn only the newest user tokens should be unseen.
    let last = hit_tokens_by_turn[config.turns_per_conversation - 1];
    let expected_prefix = 128 + 3 * (32 + 64);
    assert!(
        last >= expected_prefix - TPB,
        "expected ~{expected_prefix} hit tokens on the final turn, got {last}"
    );
}

#[test]
fn a_full_slab_refuses_rather_than_corrupting() {
    let mut store = new_store(4);
    let mut rng = SplitMix64::new(5);
    let tokens = rng.tokens(16 * 6, 32_000); // 6 blocks into 4 slots

    let block_bytes = store.layout().block_bytes();
    let lookup = store.lookup(&tokens);
    let payloads: Vec<Vec<u8>> = lookup
        .hashes
        .iter()
        .map(|&hash| trace::block_payload(hash, block_bytes))
        .collect();
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();

    let report = store.admit_sequence(&tokens, &refs);
    assert_eq!(report.inserted, 4);
    assert_eq!(report.dropped, 2);
    assert_eq!(store.resident_blocks(), 4);

    // What we stored is still correct and still usable as a prefix.
    let lookup = store.lookup(&tokens);
    assert_eq!(lookup.matched, 4);
    for &hash in &lookup.hashes[..4] {
        assert!(store.read(hash).is_some());
    }
    for &hash in &lookup.hashes[4..] {
        assert!(
            store.read(hash).is_none(),
            "must not serve a block we dropped"
        );
    }
}

#[test]
fn an_orphan_block_is_never_admitted() {
    let mut store = new_store(64);
    let mut rng = SplitMix64::new(6);
    let tokens = rng.tokens(64, 32_000);
    let block_bytes = store.layout().block_bytes();

    let hashes = store.lookup(&tokens).hashes;
    let payload = trace::block_payload(hashes[2], block_bytes);

    // Block 2 without blocks 0 and 1 is unusable.
    assert_eq!(
        store.admit(hashes[2], Some(hashes[1]), 48, &payload),
        Admit::OrphanParent
    );
    assert_eq!(store.resident_blocks(), 0);
}

#[test]
fn different_models_do_not_share_a_cache() {
    let layout = BlockLayout::tiny();
    let mut rng = SplitMix64::new(7);
    let tokens = rng.tokens(64, 32_000);

    let mut llama = KvStore::new("llama-3-8b", layout.clone(), 64).unwrap();
    let mut mistral = KvStore::new("mistral-7b", layout, 64).unwrap();

    admit_all(&mut llama, &tokens);
    assert_eq!(mistral.lookup(&tokens).matched, 0);
}
