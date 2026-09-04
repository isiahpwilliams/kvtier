//! Python bindings for the kvtier client.
//!
//! vLLM's connector is Python, so something has to cross the boundary. This
//! exposes the Rust client rather than reimplementing the wire protocol,
//! which keeps one implementation of the framing and -- more importantly --
//! one implementation of block naming. A Python reimplementation that hashed
//! even slightly differently would not fail; it would just never hit.
//!
//! Calls are synchronous. The connector's hooks are called from vLLM's
//! scheduler loop, so an embedded runtime blocking on each request is the
//! shape that fits.

use std::net::SocketAddr;

use ::kvtier::block::{BlockHash, BlockLayout, DType, PrefixHasher, TokenId};
use ::kvtier::client::KvClient;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

fn io_error(error: std::io::Error) -> PyErr {
    PyIOError::new_err(error.to_string())
}

fn parse_hash(bytes: &[u8]) -> PyResult<BlockHash> {
    let bytes: [u8; 16] = bytes
        .try_into()
        .map_err(|_| PyValueError::new_err("a block name is exactly 16 bytes"))?;
    Ok(BlockHash::from_bytes(bytes))
}

fn parse_dtype(name: &str) -> PyResult<DType> {
    Ok(match name {
        "f32" | "float32" => DType::F32,
        "f16" | "float16" | "half" => DType::F16,
        "bf16" | "bfloat16" => DType::BF16,
        "f8" | "float8" => DType::F8,
        other => return Err(PyValueError::new_err(format!("unknown dtype {other:?}"))),
    })
}

/// Names blocks exactly as the server does.
///
/// Exposed so the connector never has to reimplement the chain rule. Get it
/// subtly wrong and nothing errors -- the hit rate is just always zero.
#[pyclass]
struct Hasher {
    inner: PrefixHasher,
    layout: BlockLayout,
}

#[pymethods]
impl Hasher {
    #[new]
    #[pyo3(signature = (model_id, tokens_per_block, num_layers, num_kv_heads, head_dim, dtype))]
    fn new(
        model_id: &str,
        tokens_per_block: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: &str,
    ) -> PyResult<Self> {
        let layout = BlockLayout {
            tokens_per_block,
            num_layers,
            num_kv_heads,
            head_dim,
            dtype: parse_dtype(dtype)?,
        };
        Ok(Self {
            inner: PrefixHasher::new(model_id, &layout),
            layout,
        })
    }

    #[getter]
    fn block_bytes(&self) -> usize {
        self.layout.block_bytes()
    }

    #[getter]
    fn tokens_per_block(&self) -> usize {
        self.layout.tokens_per_block
    }

    /// Names for every *full* block of `tokens`, in order. A trailing partial
    /// block gets no name: it will grow, so its name would not stay valid.
    fn chain<'py>(&self, py: Python<'py>, tokens: Vec<TokenId>) -> Vec<Bound<'py, PyBytes>> {
        self.inner
            .chain(&tokens)
            .into_iter()
            .map(|hash| PyBytes::new(py, hash.as_bytes()))
            .collect()
    }
}

/// A connection to a kvtier server.
#[pyclass]
struct Client {
    runtime: tokio::runtime::Runtime,
    client: KvClient,
}

#[pymethods]
impl Client {
    #[new]
    fn new(address: &str) -> PyResult<Self> {
        let address: SocketAddr = address
            .parse()
            .map_err(|_| PyValueError::new_err(format!("bad address {address:?}")))?;

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(io_error)?;
        let client = runtime
            .block_on(KvClient::connect(address))
            .map_err(io_error)?;

        Ok(Self { runtime, client })
    }

    #[getter]
    fn model_id(&self) -> String {
        self.client.info().model_id.clone()
    }

    #[getter]
    fn block_bytes(&self) -> usize {
        self.client.block_bytes()
    }

    #[getter]
    fn tokens_per_block(&self) -> usize {
        self.client.info().layout.tokens_per_block
    }

    /// A `Hasher` agreeing with this server's model and layout, so names
    /// computed here are the names it stores under.
    fn hasher(&self) -> Hasher {
        let layout = self.client.info().layout.clone();
        Hasher {
            inner: PrefixHasher::new(&self.client.info().model_id, &layout),
            layout,
        }
    }

    /// How many leading blocks the server holds. Moves no KV.
    fn match_prefix(&mut self, hashes: Vec<Vec<u8>>) -> PyResult<usize> {
        let hashes = hashes
            .iter()
            .map(|bytes| parse_hash(bytes))
            .collect::<PyResult<Vec<_>>>()?;

        let client = &mut self.client;
        self.runtime
            .block_on(client.match_prefix(&hashes))
            .map_err(io_error)
    }

    /// Fetch the resident leading run. Shorter than the request means the
    /// server ran out of blocks, never that it skipped one.
    fn get_blocks<'py>(
        &mut self,
        py: Python<'py>,
        hashes: Vec<Vec<u8>>,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let hashes = hashes
            .iter()
            .map(|bytes| parse_hash(bytes))
            .collect::<PyResult<Vec<_>>>()?;

        let client = &mut self.client;
        // Release the GIL: this blocks on the network, and the connector
        // shares a process with the engine's own Python work.
        let blocks = py
            .detach(|| self.runtime.block_on(client.get_blocks(&hashes)))
            .map_err(io_error)?;

        Ok(blocks
            .into_iter()
            .map(|block| PyBytes::new(py, &block))
            .collect())
    }

    /// Offer blocks to the server. `names` pairs each block's name with the
    /// token depth it sits at; `blocks` is their payloads, concatenated.
    #[pyo3(signature = (parent, names, blocks))]
    fn put_blocks(
        &mut self,
        py: Python<'_>,
        parent: Option<Vec<u8>>,
        names: Vec<(Vec<u8>, u32)>,
        blocks: Vec<u8>,
    ) -> PyResult<(usize, usize, usize)> {
        let parent = parent.as_deref().map(parse_hash).transpose()?;
        let names = names
            .iter()
            .map(|(bytes, depth)| Ok((parse_hash(bytes)?, *depth)))
            .collect::<PyResult<Vec<_>>>()?;

        let block_bytes = self.client.block_bytes();
        if blocks.len() != names.len() * block_bytes {
            return Err(PyValueError::new_err(format!(
                "expected {} bytes for {} blocks, got {}",
                names.len() * block_bytes,
                names.len(),
                blocks.len()
            )));
        }
        let payloads: Vec<&[u8]> = blocks.chunks_exact(block_bytes).collect();

        let client = &mut self.client;
        let report = py
            .detach(|| {
                self.runtime
                    .block_on(client.put_blocks(parent, &names, &payloads))
            })
            .map_err(io_error)?;

        Ok((report.inserted, report.deduped, report.dropped))
    }

    fn stats<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let client = &mut self.client;
        let (stats, resident) = self.runtime.block_on(client.stats()).map_err(io_error)?;

        let out = PyDict::new(py);
        out.set_item("resident_blocks", resident)?;
        out.set_item("lookups", stats.lookups)?;
        out.set_item("queried_blocks", stats.queried_blocks)?;
        out.set_item("hit_blocks", stats.hit_blocks)?;
        out.set_item("inserted_blocks", stats.inserted_blocks)?;
        out.set_item("deduped_blocks", stats.deduped_blocks)?;
        out.set_item("evicted_blocks", stats.evicted_blocks)?;
        out.set_item("demoted_blocks", stats.demoted_blocks)?;
        out.set_item("blocking_demotions", stats.blocking_demotions)?;
        out.set_item("written_back", stats.written_back)?;
        out.set_item("promoted_blocks", stats.promoted_blocks)?;
        out.set_item("bytes_admitted", stats.bytes_admitted)?;
        Ok(out)
    }
}

#[pymodule]
fn kvtier(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Client>()?;
    module.add_class::<Hasher>()?;
    Ok(())
}
