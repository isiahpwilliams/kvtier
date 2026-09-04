//! Phase 1-3 demo: a synthetic multi-turn workload through the store, then
//! the same workload at shrinking cache sizes.

use kvtier::block::BlockLayout;
use kvtier::store::{KvStore, Stats};
use kvtier::trace::{self, Request, WorkloadConfig};

struct Run {
    hit_tokens: usize,
    total_tokens: usize,
    resident: usize,
    stats: Stats,
    per_turn: Vec<(usize, usize)>,
}

fn run(requests: &[Request], layout: &BlockLayout, capacity: usize, turns: usize) -> Run {
    let mut store = KvStore::new("llama-3-8b", layout.clone(), capacity).unwrap();
    let mut result = Run {
        hit_tokens: 0,
        total_tokens: 0,
        resident: 0,
        stats: Stats::default(),
        per_turn: vec![(0, 0); turns],
    };

    for request in requests {
        let lookup = store.lookup(&request.tokens);
        result.hit_tokens += lookup.hit_tokens();
        result.total_tokens += request.tokens.len();
        result.per_turn[request.turn].0 += lookup.hit_tokens();
        result.per_turn[request.turn].1 += request.tokens.len();

        // At the end of a request the engine hands back the generated KV too.
        let completed = store.lookup(&request.completed);
        let payloads: Vec<Vec<u8>> = completed
            .hashes
            .iter()
            .map(|&hash| trace::block_payload(hash, layout.block_bytes()))
            .collect();
        let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
        store.admit_sequence(&request.completed, &refs);
    }

    result.resident = store.resident_blocks();
    result.stats = store.stats();
    result
}

fn main() -> std::io::Result<()> {
    let config = WorkloadConfig::default();
    // Storage uses a tiny layout so the demo fits in a few MiB; savings are
    // priced at Llama-3-8B's real KV size.
    let layout = BlockLayout::tiny();
    let real_layout = BlockLayout::llama3_8b();
    let requests = trace::generate(&config);

    println!(
        "workload: {} conversations x {} turns, {}-token shared system prompt",
        config.conversations, config.turns_per_conversation, config.system_prompt_tokens
    );
    println!(
        "model:    llama-3-8b, {} KiB KV per token, {} MiB per {}-token block\n",
        real_layout.token_bytes() / 1024,
        real_layout.block_bytes() / (1024 * 1024),
        real_layout.tokens_per_block
    );

    let unbounded = run(&requests, &layout, 8192, config.turns_per_conversation);

    println!(
        "{:>5}  {:>12}  {:>12}  {:>9}",
        "turn", "hit tokens", "req tokens", "hit rate"
    );
    for (turn, (hit, total)) in unbounded.per_turn.iter().enumerate() {
        println!(
            "{turn:>5}  {hit:>12}  {total:>12}  {:>8.1}%",
            100.0 * *hit as f64 / *total as f64
        );
    }

    let saved = unbounded.hit_tokens;
    println!("\nblocks resident      {}", unbounded.resident);
    println!("blocks deduped       {}", unbounded.stats.deduped_blocks);
    println!(
        "tokens not prefilled {saved} of {} ({:.1}%)",
        unbounded.total_tokens,
        100.0 * saved as f64 / unbounded.total_tokens as f64
    );
    println!(
        "KV served, llama-3-8b equivalent  {:.2} GiB",
        (saved * real_layout.token_bytes()) as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // The working set is `unbounded.resident`; squeeze below it and eviction
    // has to start choosing.
    println!(
        "\ncache size vs hit rate  (working set is {} blocks)",
        unbounded.resident
    );
    println!(
        "{:>8}  {:>9}  {:>10}  {:>9}  {:>10}",
        "blocks", "of w.set", "hit rate", "evicted", "GiB served"
    );
    for fraction in [100, 75, 50, 25, 10, 5] {
        let capacity = (unbounded.resident * fraction / 100).max(8);
        let result = run(&requests, &layout, capacity, config.turns_per_conversation);
        println!(
            "{capacity:>8}  {fraction:>8}%  {:>8.1}%  {:>9}  {:>10.2}",
            100.0 * result.hit_tokens as f64 / result.total_tokens as f64,
            result.stats.evicted_blocks,
            (result.hit_tokens * real_layout.token_bytes()) as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    Ok(())
}
