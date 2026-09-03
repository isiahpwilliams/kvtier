//! Cost of the lookup path, split into its two halves.
//!
//! Every request pays this before any KV moves, so it is the floor on what
//! the tier can add to TTFT. Phase 2 puts a network in front of it; these
//! numbers are what we compare against to show the wire is the bottleneck
//! and the index is not.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use kvtier::block::{BlockLayout, PrefixHasher, TokenId};
use kvtier::store::KvStore;
use kvtier::trace::{self, SplitMix64};
use std::hint::black_box;

/// Block counts spanning a short prompt to a 16k-token context.
const BLOCK_COUNTS: [usize; 4] = [16, 64, 256, 1024];

fn tokens(blocks: usize) -> Vec<TokenId> {
    SplitMix64::new(0xBEEF).tokens(blocks * 16, 32_000)
}

/// A store holding every block of `tokens`, so lookups hit all the way down.
fn warm_store(tokens: &[TokenId]) -> KvStore {
    let layout = BlockLayout::tiny();
    let block_bytes = layout.block_bytes();
    let mut store = KvStore::new("bench", layout, tokens.len() / 16 + 1).unwrap();

    let hashes = store.lookup(tokens).hashes;
    let payloads: Vec<Vec<u8>> = hashes
        .iter()
        .map(|&hash| trace::block_payload(hash, block_bytes))
        .collect();
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    store.admit_sequence(tokens, &refs);
    store
}

/// Naming only: BLAKE3 over the token ids, no index involved.
fn bench_naming(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain");
    let hasher = PrefixHasher::new("bench", &BlockLayout::tiny());

    for blocks in BLOCK_COUNTS {
        let tokens = tokens(blocks);
        group.throughput(Throughput::Elements(blocks as u64));
        group.bench_with_input(BenchmarkId::from_parameter(blocks), &tokens, |b, tokens| {
            b.iter(|| black_box(hasher.chain(black_box(tokens))));
        });
    }
    group.finish();
}

/// Probing only: names are precomputed, so this is pure hash table walk.
fn bench_probing(c: &mut Criterion) {
    let mut group = c.benchmark_group("match_prefix");

    for blocks in BLOCK_COUNTS {
        let tokens = tokens(blocks);
        let mut store = warm_store(&tokens);
        let hashes = store.lookup(&tokens).hashes;

        group.throughput(Throughput::Elements(blocks as u64));
        group.bench_with_input(BenchmarkId::from_parameter(blocks), &hashes, |b, hashes| {
            b.iter(|| black_box(store.match_prefix(black_box(hashes))));
        });
    }
    group.finish();
}

/// The whole path a request takes: name the sequence, then probe for it.
fn bench_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_hit");

    for blocks in BLOCK_COUNTS {
        let tokens = tokens(blocks);
        let mut store = warm_store(&tokens);

        group.throughput(Throughput::Elements(blocks as u64));
        group.bench_with_input(BenchmarkId::from_parameter(blocks), &tokens, |b, tokens| {
            b.iter(|| black_box(store.lookup(black_box(tokens))));
        });
    }
    group.finish();
}

/// A cold store still pays for naming every block, but the first probe ends
/// the walk. The gap against `lookup_hit` is the cost of the probes alone.
fn bench_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_miss");

    for blocks in BLOCK_COUNTS {
        let tokens = tokens(blocks);
        let mut store = KvStore::new("bench", BlockLayout::tiny(), 16).unwrap();

        group.throughput(Throughput::Elements(blocks as u64));
        group.bench_with_input(BenchmarkId::from_parameter(blocks), &tokens, |b, tokens| {
            b.iter(|| black_box(store.lookup(black_box(tokens))));
        });
    }
    group.finish();
}

/// Serving one block's bytes: an index probe and a slice. This is what Phase
/// 2 will call once per block before handing the slice to the socket.
fn bench_read(c: &mut Criterion) {
    let tokens = tokens(256);
    let mut store = warm_store(&tokens);
    let hash = store.lookup(&tokens).hashes[128];

    c.bench_function("read_block", |b| {
        b.iter(|| black_box(store.read(black_box(hash))));
    });
}

criterion_group!(
    benches,
    bench_naming,
    bench_probing,
    bench_lookup_hit,
    bench_lookup_miss,
    bench_read
);
criterion_main!(benches);
