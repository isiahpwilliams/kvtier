//! Phase 1-3 demo: a synthetic multi-turn workload through the store, then
//! the same workload at shrinking cache sizes.

use kvtier::block::BlockLayout;
use kvtier::store::{KvStore, Stats};
use kvtier::tier::DiskTier;
use kvtier::trace::{self, Request, WorkloadConfig};

struct Run {
    hit_tokens: usize,
    total_tokens: usize,
    resident: usize,
    stats: Stats,
    disk_reads: u64,
    per_turn: Vec<(usize, usize)>,
}

fn run(
    requests: &[Request],
    layout: &BlockLayout,
    capacity: usize,
    disk_blocks: usize,
    turns: usize,
) -> Run {
    let mut store = KvStore::new("llama-3-8b", layout.clone(), capacity).unwrap();
    if disk_blocks > 0 {
        let tier = DiskTier::temporary(layout.block_bytes(), disk_blocks).unwrap();
        store = store.with_disk_tier(tier);
    }
    let mut result = Run {
        hit_tokens: 0,
        total_tokens: 0,
        resident: 0,
        stats: Stats::default(),
        disk_reads: 0,
        per_turn: vec![(0, 0); turns],
    };

    for request in requests {
        let lookup = store.lookup(&request.tokens);
        result.hit_tokens += lookup.hit_tokens();
        result.total_tokens += request.tokens.len();
        result.per_turn[request.turn].0 += lookup.hit_tokens();
        result.per_turn[request.turn].1 += request.tokens.len();

        // The engine fetches what it hit, which is what pulls demoted blocks
        // back off disk. Skipping this would make tiering look free.
        let pinned = store.pin_run(&lookup.hashes[..lookup.matched]);
        store.unpin_all(&pinned);

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
    result.disk_reads = store.disk_stats().map_or(0, |disk| disk.reads);
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

    let unbounded = run(&requests, &layout, 8192, 0, config.turns_per_conversation);

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
        "{:>7}  {:>8}  {:>8}  {:>10}  {:>8}  {:>9}  {:>9}",
        "ram", "of w.set", "hit rate", "+disk tier", "dropped", "demoted", "disk read"
    );
    for fraction in [100, 75, 50, 25, 10, 5] {
        let capacity = (unbounded.resident * fraction / 100).max(8);
        let flat = run(
            &requests,
            &layout,
            capacity,
            0,
            config.turns_per_conversation,
        );
        // Disk sized to hold the rest of the working set.
        let tiered = run(
            &requests,
            &layout,
            capacity,
            unbounded.resident,
            config.turns_per_conversation,
        );

        println!(
            "{capacity:>7}  {fraction:>7}%  {:>7.1}%  {:>9.1}%  {:>8}  {:>9}  {:>8.2} GiB",
            100.0 * flat.hit_tokens as f64 / flat.total_tokens as f64,
            100.0 * tiered.hit_tokens as f64 / tiered.total_tokens as f64,
            tiered.stats.evicted_blocks,
            tiered.stats.demoted_blocks,
            // Priced at Llama-3-8B's real block size, like everything else.
            (tiered.disk_reads * real_layout.block_bytes() as u64) as f64
                / (1024.0 * 1024.0 * 1024.0)
        );
    }

    Ok(())
}
