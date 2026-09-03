//! TCP server over a `KvStore`.
//!
//! `GetBlocks` writes block bytes from the slab straight to the socket, with
//! no staging buffer. The slice is borrowed from the slab, so the store lock
//! is held across the write and clients serialize for its duration --
//! per-block pinning is what lifts that.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

use crate::block::BlockHash;
use crate::proto::{
    HEADER_BYTES, Header, Opcode, ProtoError, Reader, ServerInfo, Writer, blocks_per_frame,
    encode_info,
};
use crate::store::KvStore;

pub struct Server {
    listener: TcpListener,
    store: Arc<Mutex<KvStore>>,
}

impl Server {
    pub async fn bind(addr: SocketAddr, store: Arc<Mutex<KvStore>>) -> io::Result<Self> {
        Ok(Self {
            listener: TcpListener::bind(addr).await?,
            store,
        })
    }

    /// The bound address. Tests bind port 0 and read the real one back.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept forever, one task per connection.
    pub async fn run(self) -> io::Result<()> {
        loop {
            let (socket, peer) = self.listener.accept().await?;
            let store = Arc::clone(&self.store);
            tokio::spawn(async move {
                if let Err(error) = serve_connection(socket, store).await {
                    eprintln!("connection {peer} ended: {error}");
                }
            });
        }
    }
}

async fn serve_connection(mut socket: TcpStream, store: Arc<Mutex<KvStore>>) -> io::Result<()> {
    // Block payloads are large and responses are already batched, so Nagle
    // buys nothing and its delay would land straight on TTFT.
    socket.set_nodelay(true)?;

    let mut payload = Vec::new();
    while let Some(header) = read_header(&mut socket).await? {
        payload.resize(header.payload_len as usize, 0);
        socket.read_exact(&mut payload).await?;

        if let Err(error) = dispatch(&mut socket, &store, header, &payload).await {
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
) -> io::Result<()> {
    match header.opcode {
        Opcode::Info => handle_info(socket, store, header).await,
        Opcode::MatchPrefix => handle_match_prefix(socket, store, header, payload).await,
        Opcode::GetBlocks => handle_get_blocks(socket, store, header, payload).await,
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
async fn handle_get_blocks(
    socket: &mut TcpStream,
    store: &Mutex<KvStore>,
    header: Header,
    payload: &[u8],
) -> io::Result<()> {
    let hashes = read_hash_request(payload)?;
    let store = store.lock().await;

    let block_bytes = store.layout().block_bytes();
    let capacity = blocks_per_frame(block_bytes);
    let mut available = 0;
    while available < hashes.len().min(capacity) && store.read(hashes[available]).is_some() {
        available += 1;
    }

    let response = Header::new(
        Opcode::GetBlocks,
        header.request_id,
        4 + available * block_bytes,
    );
    socket.write_all(&response.encode()).await?;
    socket.write_all(&(available as u32).to_be_bytes()).await?;

    // Straight from the slab to the socket. No staging buffer exists.
    for &hash in &hashes[..available] {
        let block = store.read(hash).expect("counted as resident above");
        socket.write_all(block).await?;
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
