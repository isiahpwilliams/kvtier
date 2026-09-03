//! Block identity: a block is named by its whole token history, so two
//! sequences share a name exactly when their KV bytes are identical.

use std::fmt;
use std::hash::{Hash, Hasher};

/// A token id as produced by the model's tokenizer.
pub type TokenId = u32;

/// Domain separation: a namespace digest can never be read as a block digest.
const TAG_NAMESPACE: u8 = 0x01;
const TAG_BLOCK: u8 = 0x02;

/// The name of a block: 128 bits of BLAKE3. 
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockHash([u8; 16]);

impl BlockHash {
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Hash table bucket key.
    pub fn bucket_key(&self) -> u64 {
        u64::from_le_bytes(self.0[..8].try_into().unwrap())
    }
}

impl Hash for BlockHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.bucket_key());
    }
}

impl fmt::Debug for BlockHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "blk:")?;
        for byte in &self.0[..6] {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Element type of the KV tensors; only the byte width matters to us.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    F8,
}

impl DType {
    pub const fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F8 => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F8 => "f8",
        }
    }
}

/// The physical shape of one block. Every field changes the byte layout, so
/// every field feeds into the namespace digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    /// Tokens covered by one block. vLLM's PagedAttention default is 16.
    pub tokens_per_block: usize,
    pub num_layers: usize,
    /// KV heads, not attention heads: under GQA these differ by a large factor.
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: DType,
}

impl BlockLayout {
    /// Bytes in one block. The leading 2 is K and V.
    pub fn block_bytes(&self) -> usize {
        2 * self.num_layers
            * self.tokens_per_block
            * self.num_kv_heads
            * self.head_dim
            * self.dtype.size()
    }

    /// Bytes of KV per single token, across all layers.
    pub fn token_bytes(&self) -> usize {
        self.block_bytes() / self.tokens_per_block
    }

    /// Llama-3-8B fp16: 128 KiB of KV per token, so a 32k context is 4 GiB.
    pub fn llama3_8b() -> Self {
        Self {
            tokens_per_block: 16,
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::F16,
        }
    }

    /// A deliberately tiny layout for tests: 2 KiB blocks.
    pub fn tiny() -> Self {
        Self {
            tokens_per_block: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F16,
        }
    }
}

/// Computes prefix-chained block names:
///
/// ```text
/// h_0 = H(namespace, tokens[0..16])
/// h_i = H(h_{i-1}, tokens[16i..16i+16])
/// ```
///
/// Shared prefixes therefore share names, and diverge permanently at the
/// first differing token.
pub struct PrefixHasher {
    namespace: BlockHash,
    tokens_per_block: usize,
}

impl PrefixHasher {
    pub fn new(model_id: &str, layout: &BlockLayout) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[TAG_NAMESPACE]);
        // Length-prefixed so ("ab","c") and ("a","bc") hash differently.
        hasher.update(&(model_id.len() as u64).to_le_bytes());
        hasher.update(model_id.as_bytes());
        hasher.update(&(layout.tokens_per_block as u64).to_le_bytes());
        hasher.update(&(layout.num_layers as u64).to_le_bytes());
        hasher.update(&(layout.num_kv_heads as u64).to_le_bytes());
        hasher.update(&(layout.head_dim as u64).to_le_bytes());
        hasher.update(layout.dtype.name().as_bytes());
        Self {
            namespace: truncate(hasher.finalize()),
            tokens_per_block: layout.tokens_per_block,
        }
    }

    /// Synthetic root: parent of every first block, never stored.
    pub fn namespace(&self) -> BlockHash {
        self.namespace
    }

    pub fn tokens_per_block(&self) -> usize {
        self.tokens_per_block
    }

    /// One link of the chain.
    pub fn child(&self, parent: BlockHash, tokens: &[TokenId]) -> BlockHash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[TAG_BLOCK]);
        hasher.update(parent.as_bytes());
        hasher.update(&(tokens.len() as u64).to_le_bytes());
        for token in tokens {
            hasher.update(&token.to_le_bytes());
        }
        truncate(hasher.finalize())
    }

    /// Names for every *full* block, in order. `chunks_exact` drops the
    /// trailing partial block: it will grow, so its name would not stay valid.
    pub fn chain(&self, tokens: &[TokenId]) -> Vec<BlockHash> {
        let mut out = Vec::with_capacity(tokens.len() / self.tokens_per_block);
        let mut parent = self.namespace;
        for chunk in tokens.chunks_exact(self.tokens_per_block) {
            let hash = self.child(parent, chunk);
            out.push(hash);
            parent = hash;
        }
        out
    }
}

fn truncate(hash: blake3::Hash) -> BlockHash {
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    BlockHash(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: usize) -> Vec<TokenId> {
        (0..n as TokenId).collect()
    }

    #[test]
    fn layout_sizes_match_reality() {
        let layout = BlockLayout::llama3_8b();
        assert_eq!(layout.token_bytes(), 128 * 1024);
        assert_eq!(layout.block_bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn hashing_is_deterministic() {
        let layout = BlockLayout::tiny();
        let a = PrefixHasher::new("llama-3-8b", &layout);
        let b = PrefixHasher::new("llama-3-8b", &layout);
        assert_eq!(a.chain(&seq(64)), b.chain(&seq(64)));
    }

    #[test]
    fn shared_prefix_yields_shared_names() {
        let layout = BlockLayout::tiny();
        let hasher = PrefixHasher::new("m", &layout);

        // Two sequences agreeing for 48 tokens (3 blocks) then diverging.
        let mut left = seq(48);
        let mut right = seq(48);
        left.extend([900, 901, 902, 903].repeat(4)); // 16 more
        right.extend([500, 501, 502, 503].repeat(4));

        let lh = hasher.chain(&left);
        let rh = hasher.chain(&right);
        assert_eq!(lh.len(), 4);
        assert_eq!(lh[..3], rh[..3], "shared prefix must share names");
        assert_ne!(lh[3], rh[3], "divergence must break the chain");
    }

    #[test]
    fn position_matters() {
        // The same tokens at a different offset must get a different name.
        let layout = BlockLayout::tiny();
        let hasher = PrefixHasher::new("m", &layout);
        let block: Vec<TokenId> = (0..16).collect();

        let first = hasher.chain(&block);
        let mut shifted = vec![777; 16];
        shifted.extend(&block);
        let second = hasher.chain(&shifted);

        assert_ne!(first[0], second[1]);
    }

    #[test]
    fn different_models_never_collide() {
        let layout = BlockLayout::tiny();
        let a = PrefixHasher::new("llama-3-8b", &layout);
        let b = PrefixHasher::new("mistral-7b", &layout);
        assert_ne!(a.chain(&seq(32)), b.chain(&seq(32)));
    }

    #[test]
    fn different_layouts_never_collide() {
        let a = PrefixHasher::new("m", &BlockLayout::tiny());
        let mut other = BlockLayout::tiny();
        other.dtype = DType::F8;
        let b = PrefixHasher::new("m", &other);
        assert_ne!(a.chain(&seq(32)), b.chain(&seq(32)));
    }

    #[test]
    fn partial_trailing_block_is_not_named() {
        let layout = BlockLayout::tiny();
        let hasher = PrefixHasher::new("m", &layout);
        // 40 tokens = 2 full blocks + 8 leftover.
        assert_eq!(hasher.chain(&seq(40)).len(), 2);
        // Names stay stable as the tail grows.
        assert_eq!(hasher.chain(&seq(40)), hasher.chain(&seq(47))[..2]);
    }
}
