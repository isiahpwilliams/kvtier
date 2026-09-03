//! Phase 1 demo: run a synthetic multi-turn workload through the store.

use kvtier::block::BlockLayout;
use kvtier::store::KvStore;
use kvtier::trace::{self, WorkloadConfig};

fn main() -> std::io::Result<()> {
    let config = WorkloadConfig::default();
    // Storage uses a tiny layout so the demo fits in a few MiB; savings are
    // priced at Llama-3-8B's real KV size.
    let layout = BlockLayout::tiny();
    let real_layout = BlockLayout::llama3_8b();
    let mut store = KvStore::new("llama-3-8b", layout.clone(), 8192)?;

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

    let mut prefilled_tokens = 0usize;
    let mut total_tokens = 0usize;
    let mut per_turn = vec![(0usize, 0usize); config.turns_per_conversation];

    for request in &requests {
        let lookup = store.lookup(&request.tokens);

        // Tokens the engine would still have to prefill.
        let missing = request.tokens.len() - lookup.hit_tokens();
        prefilled_tokens += missing;
        total_tokens += request.tokens.len();

        let slot = &mut per_turn[request.turn];
        slot.0 += lookup.hit_tokens();
        slot.1 += request.tokens.len();

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

    println!(
        "{:>5}  {:>12}  {:>12}  {:>9}",
        "turn", "hit tokens", "req tokens", "hit rate"
    );
    for (turn, (hit, total)) in per_turn.iter().enumerate() {
        println!(
            "{turn:>5}  {hit:>12}  {total:>12}  {:>8.1}%",
            100.0 * *hit as f64 / *total as f64
        );
    }

    let stats = store.stats();
    let saved_tokens = total_tokens - prefilled_tokens;
    println!("\nblocks resident      {}", store.resident_blocks());
    println!("blocks deduped       {}", stats.deduped_blocks);
    println!("block hit rate       {:.1}%", 100.0 * stats.hit_rate());
    println!(
        "tokens not prefilled {saved_tokens} of {total_tokens} ({:.1}%)",
        100.0 * saved_tokens as f64 / total_tokens as f64
    );
    println!(
        "KV served, llama-3-8b equivalent  {:.2} GiB",
        (saved_tokens * real_layout.token_bytes()) as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "KV stored, llama-3-8b equivalent  {:.2} GiB",
        (store.resident_blocks() * real_layout.block_bytes()) as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    Ok(())
}
