//! Synthetic workloads: many conversations sharing one long system prompt,
//! each growing a turn at a time, interleaved as a real server sees them.
//! In the library because every later phase benchmarks against it.

use crate::block::{BlockHash, TokenId};

/// Small, fast, and uniform enough that token ids do not accidentally repeat
/// and inflate the hit rate.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn token(&mut self, vocab: u32) -> TokenId {
        (self.next_u64() % u64::from(vocab)) as TokenId
    }

    pub fn tokens(&mut self, count: usize, vocab: u32) -> Vec<TokenId> {
        (0..count).map(|_| self.token(vocab)).collect()
    }
}

#[derive(Clone, Debug)]
pub struct WorkloadConfig {
    /// Shared across every conversation: the prefix we expect to dedup.
    pub system_prompt_tokens: usize,
    pub conversations: usize,
    pub turns_per_conversation: usize,
    pub user_turn_tokens: usize,
    pub reply_tokens: usize,
    pub vocab: u32,
    pub seed: u64,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            system_prompt_tokens: 512,
            conversations: 8,
            turns_per_conversation: 4,
            user_turn_tokens: 64,
            reply_tokens: 128,
            vocab: 32_000,
            seed: 0xC0FFEE,
        }
    }
}

/// Two sequences, because the cache sees a request at two moments: `tokens`
/// is the prompt we are *asked* about at prefill, `completed` adds the tokens
/// the model generated, which we are *given* at the end. Decode KV is just as
/// reusable, and dropping it costs a turn of hit length.
#[derive(Clone, Debug)]
pub struct Request {
    pub conversation: usize,
    pub turn: usize,
    pub tokens: Vec<TokenId>,
    pub completed: Vec<TokenId>,
}

/// Round-robin across conversations, so every turn 1 arrives before any
/// turn 2 -- the interleaving that stresses the cache.
pub fn generate(config: &WorkloadConfig) -> Vec<Request> {
    let mut rng = SplitMix64::new(config.seed);
    let system_prompt = rng.tokens(config.system_prompt_tokens, config.vocab);

    let mut histories: Vec<Vec<TokenId>> = vec![system_prompt.clone(); config.conversations];
    let mut requests = Vec::new();

    for turn in 0..config.turns_per_conversation {
        for (conversation, history) in histories.iter_mut().enumerate() {
            history.extend(rng.tokens(config.user_turn_tokens, config.vocab));
            let prompt = history.clone();
            // The model's own reply becomes part of the prefix for next turn.
            history.extend(rng.tokens(config.reply_tokens, config.vocab));
            requests.push(Request {
                conversation,
                turn,
                tokens: prompt,
                completed: history.clone(),
            });
        }
    }

    requests
}

/// Deterministic stand-in for KV bytes, derived from the block's name so a
/// read-back can be checked.
pub fn fill_block(dst: &mut [u8], hash: BlockHash) {
    let mut rng = SplitMix64::new(hash.bucket_key());
    for chunk in dst.chunks_mut(8) {
        let word = rng.next_u64().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
}

pub fn block_payload(hash: BlockHash, block_bytes: usize) -> Vec<u8> {
    let mut buffer = vec![0u8; block_bytes];
    fill_block(&mut buffer, hash);
    buffer
}
