//! TCP server over a `KvStore`.
//!
//! `GetBlocks` writes block bytes from the slab straight to the socket, with
//! no staging buffer. Pinning is what lets that run unlocked: the store lock
//! is taken only to reserve the run and again to release it.
//!
//! Demoted blocks are read back the same way. The reads are issued in
//! parallel on the blocking pool, since `MatchPrefix` tells us the whole run
//! up front -- there is nothing to predict, so nothing is speculative.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

use crate::block::BlockHash;
use crate::proto::{
    HEADER_BYTES, Header, Opcode, ProtoError, Reader, ServerInfo, Writer, blocks_per_frame,
    encode_info,
};
use crate::slab::PinnedBlock;
use crate::store::{FetchPart, KvStore, Writeback};
use crate::tier::{DiskReader, DiskWriter};

/// How often the writeback loop looks for dirty blocks to copy to disk.
pub const DEFAULT_WRITEBACK_INTERVAL: Duration = Duration::from_millis(5);
/// Blocks copied per round. Enough to stay ahead of admissions without
/// holding the lock long while it picks them.
pub const DEFAULT_WRITEBACK_BATCH: usize = 32;
/// RAM utilization above which writeback runs. Below it there is no eviction
/// pressure, so cleaning blocks would be work nobody needs.
pub const DEFAULT_WRITEBACK_WATERMARK: f64 = 0.5;

pub struct Server {
    listener: TcpListener,
    store: Arc<Mutex<KvStore>>,
    parallel_reads: usize,
    writeback: Option<WritebackConfig>,
}

/// Tuning for the background writeback loop.
#[derive(Clone, Copy, Debug)]
pub struct WritebackConfig {
    pub interval: Duration,
    pub batch: usize,
    pub watermark: f64,
}

impl Default for WritebackConfig {
    fn default() -> Self {
        Self {
            interval: DEFAULT_WRITEBACK_INTERVAL,
            batch: DEFAULT_WRITEBACK_BATCH,
            watermark: DEFAULT_WRITEBACK_WATERMARK,
        }
    }
}

impl Server {
    pub async fn bind(addr: SocketAddr, store: Arc<Mutex<KvStore>>) -> io::Result<Self> {
        Ok(Self {
            listener: TcpListener::bind(addr).await?,
            store,
            parallel_reads: DEFAULT_PARALLEL_READS,
            writeback: Some(WritebackConfig::default()),
        })
    }

    /// Tune, or with `None` disable, background writeback. Disabled, every
    /// demotion pays for its own write with the store lock held.
    pub fn with_writeback(mut self, writeback: Option<WritebackConfig>) -> Self {
        self.writeback = writeback;
        self
    }

    /// How many disk reads to have in flight per fetch. Higher gives the
    /// drive more to overlap; 1 makes promotions strictly sequential.
    pub fn with_parallel_reads(mut self, reads: usize) -> Self {
        self.parallel_reads = reads.max(1);
        self
    }

    /// The bound address. Tests bind port 0 and read the real one back.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept forever, one task per connection.
    pub async fn run(self) -> io::Result<()> {
        if let Some(config) = self.writeback {
            tokio::spawn(writeback_loop(Arc::clone(&self.store), config));
        }

        loop {
            let (socket, peer) = self.listener.accept().await?;
            let store = Arc::clone(&self.store);
            let parallel_reads = self.parallel_reads;
            tokio::spawn(async move {
                if let Err(error) = serve_connection(socket, store, parallel_reads).await {
                    eprintln!("connection {peer} ended: {error}");
                }
            });
        }
    }
}

async fn serve_connection(
    mut socket: TcpStream,
    store: Arc<Mutex<KvStore>>,
    parallel_reads: usize,
) -> io::Result<()> {
    // Block payloads are large and responses are already batched, so Nagle
    // buys nothing and its delay would land straight on TTFT.
    socket.set_nodelay(true)?;

    let mut payload = Vec::new();
    while let Some(header) = read_header(&mut socket).await? {
        payload.resize(header.payload_len as usize, 0);
        socket.read_exact(&mut payload).await?;

        if let Err(error) = dispatch(&mut socket, &store, header, &payload, parallel_reads).await {
            // A protocol error is the peer's fault and is recoverable: report
            // it and keep the connection. An I/O error is not.
            match error.downcast::<ProtoError>() {
                Ok(proto_error) => {
                    write_error(&mut socket, header.request_id, &proto_error).await?
                }
                Err(io_error) => return Err(io_error),
            }
        }
    }
    Ok(())
}

/// `Ok(None)` on a clean EOF between frames, which is how a client hangs up.
async fn read_header(socket: &mut TcpStream) -> io::Result<Option<Header>> {
    let mut bytes = [0u8; HEADER_BYTES];
    match socket.read_exact(&mut bytes).await {
        Ok(_) => Ok(Some(Header::decode(&bytes)?)),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(error) => Err(error),
    }
}

async fn dispatch(
    socket: &mut TcpStream,
    store: &Mutex<KvStore>,
    header: Header,
    payload: &[u8],
    parallel_reads: usize,
) -> io::Result<()> {
    match header.opcode {
        Opcode::Info => handle_info(socket, store, header).await,
        Opcode::MatchPrefix => handle_match_prefix(socket, store, header, payload).await,
        Opcode::GetBlocks => {
            handle_get_blocks(socket, store, header, payload, parallel_reads).await
        }
        Opcode::PutBlocks => handle_put_blocks(socket, store, header, payload).await,
        Opcode::Stats => handle_stats(socket, store, header).await,
        Opcode::Error => Err(ProtoError::BadOpcode(Opcode::Error as u8).into()),
    }
}

async fn handle_info(
    socket: &mut TcpStream,
    store: &Mutex<KvStore>,
    header: Header,
) -> io::Result<()> {
    let info = {
        let store = store.lock().await;
        ServerInfo {
            model_id: store.model_id().to_string(),
            layout: store.layout().clone(),
        }
    };
    respond(socket, header, &encode_info(&info)).await
}

async fn handle_match_prefix(
    socket: &mut TcpStream,
    store: &Mutex<KvStore>,
    header: Header,
    payload: &[u8],
) -> io::Result<()> {
    let hashes = read_hash_request(payload)?;
    let matched = store.lock().await.match_prefix(&hashes);

    let mut writer = Writer::new();
    writer.u32(matched as u32);
    respond(socket, header, &writer.finish()).await
}

/// Returns the resident *leading run*, not whichever names happen to be
/// present: a block past a gap is unusable, so sending it wastes bandwidth.
/// Truncated to one frame, which the client just asks past.
///
/// The store lock is held only long enough to pin the run. The write itself
/// runs unlocked, so a large transfer no longer blocks other clients.
async fn handle_get_blocks(
    socket: &mut TcpStream,
    store: &Mutex<KvStore>,
    header: Header,
    payload: &[u8],
    parallel_reads: usize,
) -> io::Result<()> {
    let hashes = read_hash_request(payload)?;

    // Reserve the run, then let go of the lock for the slow parts.
    let (parts, reader, block_bytes) = {
        let mut store = store.lock().await;
        let block_bytes = store.layout().block_bytes();
        let capacity = blocks_per_frame(block_bytes).min(hashes.len());
        (
            store.begin_fetch(&hashes[..capacity]),
            store.disk_reader(),
            block_bytes,
        )
    };

    let parts = fill_from_disk(parts, reader, parallel_reads).await;
    let pinned = store.lock().await.finish_fetch(parts);

    let result = write_blocks(socket, header, &pinned, block_bytes).await;

    // Unconditional: a pin dropped on an error path would make its block
    // unevictable for the life of the process.
    store.lock().await.unpin_all(&pinned);
    result
}

/// Enough queue depth for the drive to overlap requests, without handing the
/// blocking pool one thread per block.
pub const DEFAULT_PARALLEL_READS: usize = 8;

/// Fill every demoted block in the run, off the store lock and in parallel.
///
/// Stripes rather than one task per block: a long run would otherwise spawn
/// hundreds of blocking tasks to do work the drive cannot overlap anyway.
async fn fill_from_disk(
    parts: Vec<FetchPart>,
    reader: Option<DiskReader>,
    parallel_reads: usize,
) -> Vec<FetchPart> {
    let Some(reader) = reader else {
        return parts;
    };
    let pending = parts
        .iter()
        .filter(|part| matches!(part, FetchPart::Pending(_)))
        .count();
    if pending == 0 {
        return parts;
    }

    let stripe = parts.len().div_ceil(parallel_reads).max(1);
    let mut tasks = Vec::new();
    let mut rest = parts;
    while !rest.is_empty() {
        let tail = rest.split_off(stripe.min(rest.len()));
        let mut chunk = std::mem::replace(&mut rest, tail);
        let reader = reader.clone();

        tasks.push(tokio::task::spawn_blocking(move || {
            for part in &mut chunk {
                if let FetchPart::Pending(promotion) = part {
                    // A failed read leaves the block unfilled; `finish_fetch`
                    // truncates the run there rather than serving a hole.
                    let _ = promotion.fill(&reader);
                }
            }
            chunk
        }));
    }

    let mut filled = Vec::new();
    for task in tasks {
        match task.await {
            Ok(chunk) => filled.extend(chunk),
            // Only reachable if the runtime is shutting down under us.
            Err(_) => break,
        }
    }
    filled
}

async fn write_blocks(
    socket: &mut TcpStream,
    header: Header,
    pinned: &[PinnedBlock],
    block_bytes: usize,
) -> io::Result<()> {
    let response = Header::new(
        Opcode::GetBlocks,
        header.request_id,
        4 + pinned.len() * block_bytes,
    );
    socket.write_all(&response.encode()).await?;
    socket
        .write_all(&(pinned.len() as u32).to_be_bytes())
        .await?;

    // Straight from the slab to the socket. No staging buffer exists.
    for block in pinned {
        socket.write_all(block.bytes()).await?;
    }
    Ok(())
}

async fn handle_put_blocks(
    socket: &mut TcpStream,
    store: &Mutex<KvStore>,
    header: Header,
    payload: &[u8],
) -> io::Result<()> {
    let mut reader = Reader::new(payload);

    let parent = match reader.u8()? {
        0 => None,
        _ => Some(reader.hash()?),
    };
    let count = reader.block_count()?;
    let names: Vec<(BlockHash, u32)> = (0..count)
        .map(|_| Ok((reader.hash()?, reader.u32()?)))
        .collect::<Result<_, ProtoError>>()?;

    let mut store = store.lock().await;
    let block_bytes = store.layout().block_bytes();
    let blocks: Vec<&[u8]> = (0..count)
        .map(|_| reader.bytes(block_bytes))
        .collect::<Result<_, ProtoError>>()?;
    reader.finish()?;

    let report = store.admit_chain(parent, &names, &blocks);
    drop(store);

    let mut writer = Writer::new();
    writer
        .u32(report.inserted as u32)
        .u32(report.deduped as u32)
        .u32(report.dropped as u32);
    respond(socket, header, &writer.finish()).await
}

async fn handle_stats(
    socket: &mut TcpStream,
    store: &Mutex<KvStore>,
    header: Header,
) -> io::Result<()> {
    let (stats, resident) = {
        let store = store.lock().await;
        (store.stats(), store.resident_blocks())
    };

    let mut writer = Writer::new();
    writer
        .u64(stats.lookups)
        .u64(stats.queried_blocks)
        .u64(stats.hit_blocks)
        .u64(stats.inserted_blocks)
        .u64(stats.deduped_blocks)
        .u64(stats.rejected_blocks)
        .u64(stats.evicted_blocks)
        .u64(stats.demoted_blocks)
        .u64(stats.blocking_demotions)
        .u64(stats.written_back)
        .u64(stats.promoted_blocks)
        .u64(stats.bytes_admitted)
        .u64(resident as u64);
    respond(socket, header, &writer.finish()).await
}

fn read_hash_request(payload: &[u8]) -> Result<Vec<BlockHash>, ProtoError> {
    let mut reader = Reader::new(payload);
    let hashes = reader.hashes()?;
    reader.finish()?;
    Ok(hashes)
}

/// Header and payload in one write, so a small response is one syscall and
/// one segment rather than two.
async fn respond(socket: &mut TcpStream, request: Header, payload: &[u8]) -> io::Result<()> {
    let header = Header::new(request.opcode, request.request_id, payload.len());
    let mut frame = Vec::with_capacity(HEADER_BYTES + payload.len());
    frame.extend_from_slice(&header.encode());
    frame.extend_from_slice(payload);
    socket.write_all(&frame).await
}

async fn write_error(
    socket: &mut TcpStream,
    request_id: u32,
    error: &ProtoError,
) -> io::Result<()> {
    let mut writer = Writer::new();
    writer.string(&error.to_string());
    let payload = writer.finish();

    let header = Header::new(Opcode::Error, request_id, payload.len());
    let mut frame = Vec::with_capacity(HEADER_BYTES + payload.len());
    frame.extend_from_slice(&header.encode());
    frame.extend_from_slice(&payload);
    socket.write_all(&frame).await
}

/// Copy dirty blocks to disk ahead of the eviction that will want them gone.
///
/// A block with a disk copy costs nothing to evict, so this converts what
/// would be a blocking write on the admit path into a background one. It only
/// runs under memory pressure: with RAM to spare there is no eviction coming,
/// and cleaning blocks would be work nobody needs.
async fn writeback_loop(store: Arc<Mutex<KvStore>>, config: WritebackConfig) {
    loop {
        tokio::time::sleep(config.interval).await;

        let (jobs, writer) = {
            let mut store = store.lock().await;
            if store.utilization() < config.watermark {
                continue;
            }
            // A job holds a pin, and pinned blocks cannot be evicted. Taking
            // too large a share of RAM would starve the fetch path, which
            // needs a free slot per block it promotes.
            let budget = config.batch.min(store.ram_capacity() / 4).max(1);
            (store.begin_writeback(budget), store.disk_writer())
        };
        if jobs.is_empty() {
            continue;
        }

        let jobs = flush_to_disk(jobs, writer).await;
        store.lock().await.finish_writeback(jobs);
    }
}

/// Write the batch off the lock, striped the same way reads are.
async fn flush_to_disk(jobs: Vec<Writeback>, writer: Option<DiskWriter>) -> Vec<Writeback> {
    let Some(writer) = writer else {
        return jobs;
    };

    let stripe = jobs.len().div_ceil(DEFAULT_PARALLEL_READS).max(1);
    let mut tasks = Vec::new();
    let mut rest = jobs;
    while !rest.is_empty() {
        let tail = rest.split_off(stripe.min(rest.len()));
        let mut chunk = std::mem::replace(&mut rest, tail);
        let writer = writer.clone();

        tasks.push(tokio::task::spawn_blocking(move || {
            for job in &mut chunk {
                // A failed write leaves the block dirty, which is safe: it
                // just means the next eviction pays for it.
                let _ = job.flush(&writer);
            }
            chunk
        }));
    }

    let mut done = Vec::new();
    for task in tasks {
        match task.await {
            Ok(chunk) => done.extend(chunk),
            Err(_) => break,
        }
    }
    done
}
