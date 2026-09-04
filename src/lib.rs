//! kvtier -- a KV cache tier for LLM inference engines.
//!
//! Phase 1: single-node, in-memory, no eviction. Names blocks by their token
//! prefix, stores them in a slab, and answers "how much of this sequence have
//! we already computed?".

pub mod block;
pub mod client;
pub mod evict;
pub mod index;
pub mod proto;
pub mod server;
pub mod slab;
pub mod store;
pub mod trace;

pub use block::{BlockHash, BlockLayout, DType, PrefixHasher, TokenId};
pub use index::{Entry, Index, IndexError};
pub use slab::{Slab, SlotId};
pub use store::{Admit, AdmitReport, KvStore, Lookup, Stats};
pub use trace::{Request, WorkloadConfig};
