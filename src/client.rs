//! Client for a `kvtier` server.
//!
//! One connection, requests issued in order. `connect` performs an `Info`
//! exchange first: block names only mean anything under a matching layout, so
//! a mismatch is caught at connect time rather than as corrupt KV later.

use std::io;
use std::net::SocketAddr;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::block::{BlockHash, BlockLayout};
use crate::proto::{
    HEADER_BYTES, Header, Opcode, ProtoError, Reader, ServerInfo, Writer, blocks_per_frame,
    decode_info,
};
use crate::store::{AdmitReport, Stats};

pub struct KvClient {
    socket: TcpStream,
    info: ServerInfo,
    next_request_id: u32,
}

impl KvClient {
    pub async fn connect(addr: SocketAddr) -> io::Result<Self> {
        let socket = TcpStream::connect(addr).await?;
        socket.set_nodelay(true)?;

        let mut client = Self {
            socket,
            info: ServerInfo {
                model_id: String::new(),
                layout: BlockLayout::tiny(),
            },
            next_request_id: 1,
        };
        client.info = client.request_info().await?;
        Ok(client)
    }

    /// Connect and refuse the server if its layout is not the one we name
    /// blocks under.
    pub async fn connect_expecting(addr: SocketAddr, layout: &BlockLayout) -> io::Result<Self> {
        let client = Self::connect(addr).await?;
        if &client.info.layout != layout {
            return Err(ProtoError::LayoutMismatch.into());
        }
        Ok(client)
    }

    pub fn info(&self) -> &ServerInfo {
        &self.info
    }

    pub fn block_bytes(&self) -> usize {
        self.info.layout.block_bytes()
    }

    /// How many leading blocks the server holds. Cheap: no KV moves.
    pub async fn match_prefix(&mut self, hashes: &[BlockHash]) -> io::Result<usize> {
        let mut writer = Writer::new();
        writer.hashes(hashes);

        let payload = self
            .round_trip(Opcode::MatchPrefix, &writer.finish())
            .await?;
        let mut reader = Reader::new(&payload);
        let matched = reader.u32()? as usize;
        reader.finish()?;
        Ok(matched)
    }

    /// Fetch the resident leading run of `hashes`: possibly shorter than the
    /// request, never gapped. A run past one frame takes several round trips,
    /// which is safe because the run always starts at the request's first
    /// hash.
    pub async fn get_blocks(&mut self, hashes: &[BlockHash]) -> io::Result<Vec<Vec<u8>>> {
        let per_frame = blocks_per_frame(self.block_bytes());
        let mut blocks = Vec::new();

        while blocks.len() < hashes.len() {
            let want = &hashes[blocks.len()..];
            let batch = self
                .get_one_frame(&want[..want.len().min(per_frame)])
                .await?;

            let short = batch.len() < per_frame;
            blocks.extend(batch);
            // A frame the server did not fill means it ran out of resident
            // blocks, not out of frame -- so there is nothing more to ask for.
            if short {
                break;
            }
        }
        Ok(blocks)
    }

    /// Same run as `get_blocks`, written straight into `out` instead of into
    /// a fresh allocation per block. Returns how many blocks were written.
    ///
    /// The caller owns the destination, so KV can land in page-locked memory
    /// the GPU will DMA out of, rather than in a heap buffer that has to be
    /// copied again on its way there.
    pub async fn get_blocks_into(
        &mut self,
        hashes: &[BlockHash],
        out: &mut [u8],
    ) -> io::Result<usize> {
        let block_bytes = self.block_bytes();
        let capacity = out.len() / block_bytes;
        let hashes = &hashes[..hashes.len().min(capacity)];

        let per_frame = blocks_per_frame(block_bytes);
        let mut done = 0;

        while done < hashes.len() {
            let want = &hashes[done..];
            let want = &want[..want.len().min(per_frame)];

            let count = self
                .get_frame_into(want, &mut out[done * block_bytes..])
                .await?;
            done += count;

            // A frame the server did not fill means it ran out of resident
            // blocks, not out of frame.
            if count < want.len() {
                break;
            }
        }
        Ok(done)
    }

    /// One frame of blocks, read straight from the socket into `out`.
    ///
    /// `round_trip` stages the whole response in a heap buffer and the caller
    /// copies it out again. For a fetch that buffer is the KV itself, so both
    /// the allocation and the copy are worth removing. Kept separate from
    /// `round_trip` rather than sharing helpers with it: pulling the header
    /// read out into a shared function measurably slowed the generic path.
    async fn get_frame_into(&mut self, hashes: &[BlockHash], out: &mut [u8]) -> io::Result<usize> {
        let block_bytes = self.block_bytes();

        let mut writer = Writer::new();
        writer.hashes(hashes);
        let request_id = self.send(Opcode::GetBlocks, &writer.finish()).await?;

        let mut header_bytes = [0u8; HEADER_BYTES];
        self.socket.read_exact(&mut header_bytes).await?;
        let response = Header::decode(&header_bytes)?;
        if response.request_id != request_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "response id {} for request {request_id}",
                    response.request_id
                ),
            ));
        }
        if response.opcode == Opcode::Error {
            let mut body = vec![0u8; response.payload_len as usize];
            self.socket.read_exact(&mut body).await?;
            let message = Reader::new(&body).string()?;
            return Err(io::Error::new(io::ErrorKind::InvalidInput, message));
        }

        let mut count_bytes = [0u8; 4];
        self.socket.read_exact(&mut count_bytes).await?;
        let count = u32::from_be_bytes(count_bytes) as usize;

        if count > hashes.len() || response.payload_len as usize != 4 + count * block_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{count} blocks in a {} byte frame for {} names",
                    response.payload_len,
                    hashes.len()
                ),
            ));
        }
        self.socket
            .read_exact(&mut out[..count * block_bytes])
            .await?;
        Ok(count)
    }

    async fn get_one_frame(&mut self, hashes: &[BlockHash]) -> io::Result<Vec<Vec<u8>>> {
        let mut writer = Writer::new();
        writer.hashes(hashes);

        let payload = self.round_trip(Opcode::GetBlocks, &writer.finish()).await?;
        let block_bytes = self.block_bytes();

        let mut reader = Reader::new(&payload);
        let count = reader.block_count()?;
        let blocks = (0..count)
            .map(|_| Ok(reader.bytes(block_bytes)?.to_vec()))
            .collect::<Result<Vec<_>, ProtoError>>()?;
        reader.finish()?;
        Ok(blocks)
    }

    /// Offer blocks to the server. `parent` is what the first one attaches
    /// to; `None` starts a sequence. Split across frames when needed, each
    /// chunk parented on the last block of the one before it.
    pub async fn put_blocks(
        &mut self,
        parent: Option<BlockHash>,
        names: &[(BlockHash, u32)],
        blocks: &[&[u8]],
    ) -> io::Result<AdmitReport> {
        assert_eq!(names.len(), blocks.len(), "one name per payload");

        let per_frame = blocks_per_frame(self.block_bytes());
        let mut total = AdmitReport::default();
        let mut parent = parent;
        let mut remaining = names.len();

        for (names, blocks) in names.chunks(per_frame).zip(blocks.chunks(per_frame)) {
            let report = self.put_one_frame(parent, names, blocks).await?;
            total.inserted += report.inserted;
            total.deduped += report.deduped;
            total.dropped += report.dropped;
            remaining -= names.len();

            if report.dropped > 0 {
                total.dropped += remaining;
                break;
            }
            parent = names.last().map(|&(hash, _)| hash).or(parent);
        }
        Ok(total)
    }

    async fn put_one_frame(
        &mut self,
        parent: Option<BlockHash>,
        names: &[(BlockHash, u32)],
        blocks: &[&[u8]],
    ) -> io::Result<AdmitReport> {
        let mut writer = Writer::new();
        match parent {
            Some(parent) => {
                writer.u8(1).hash(parent);
            }
            None => {
                writer.u8(0);
            }
        }
        writer.u32(names.len() as u32);
        for &(hash, depth) in names {
            writer.hash(hash).u32(depth);
        }
        for block in blocks {
            writer.bytes(block);
        }

        let payload = self.round_trip(Opcode::PutBlocks, &writer.finish()).await?;
        let mut reader = Reader::new(&payload);
        let report = AdmitReport {
            inserted: reader.u32()? as usize,
            deduped: reader.u32()? as usize,
            dropped: reader.u32()? as usize,
        };
        reader.finish()?;
        Ok(report)
    }

    pub async fn stats(&mut self) -> io::Result<(Stats, usize)> {
        let payload = self.round_trip(Opcode::Stats, &[]).await?;
        let mut reader = Reader::new(&payload);
        let stats = Stats {
            lookups: reader.u64()?,
            queried_blocks: reader.u64()?,
            hit_blocks: reader.u64()?,
            inserted_blocks: reader.u64()?,
            deduped_blocks: reader.u64()?,
            rejected_blocks: reader.u64()?,
            evicted_blocks: reader.u64()?,
            demoted_blocks: reader.u64()?,
            blocking_demotions: reader.u64()?,
            written_back: reader.u64()?,
            promoted_blocks: reader.u64()?,
            bytes_admitted: reader.u64()?,
        };
        let resident = reader.u64()? as usize;
        reader.finish()?;
        Ok((stats, resident))
    }

    async fn request_info(&mut self) -> io::Result<ServerInfo> {
        let payload = self.round_trip(Opcode::Info, &[]).await?;
        Ok(decode_info(&payload)?)
    }

    async fn send(&mut self, opcode: Opcode, payload: &[u8]) -> io::Result<u32> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);

        let header = Header::new(opcode, request_id, payload.len());
        let mut frame = Vec::with_capacity(HEADER_BYTES + payload.len());
        frame.extend_from_slice(&header.encode());
        frame.extend_from_slice(payload);
        self.socket.write_all(&frame).await?;
        Ok(request_id)
    }

    async fn round_trip(&mut self, opcode: Opcode, payload: &[u8]) -> io::Result<Vec<u8>> {
        let request_id = self.send(opcode, payload).await?;

        let mut header_bytes = [0u8; HEADER_BYTES];
        self.socket.read_exact(&mut header_bytes).await?;
        let response = Header::decode(&header_bytes)?;

        let mut body = vec![0u8; response.payload_len as usize];
        self.socket.read_exact(&mut body).await?;

        if response.request_id != request_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "response id {} for request {request_id}",
                    response.request_id
                ),
            ));
        }
        if response.opcode == Opcode::Error {
            let message = Reader::new(&body).string()?;
            return Err(io::Error::new(io::ErrorKind::InvalidInput, message));
        }
        Ok(body)
    }
}
