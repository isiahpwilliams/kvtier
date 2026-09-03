//! Wire format. Pure encoding, no I/O, so it can be tested without a socket.
//!
//! Every message is a 16-byte header followed by a payload. Requests and
//! responses share the header; a response echoes the request's opcode and id,
//! or comes back as `Error`.
//!
//! ```text
//! 0       4    5     6     7     8            12           16
//! | magic | ver | op  | flg | rsv | request_id | payload_len |
//! ```

use crate::block::{BlockHash, BlockLayout, DType};

/// "KVT1". Present on every frame so a desynced stream fails loudly on the
/// next header rather than reading a length out of block payload bytes.
pub const MAGIC: u32 = 0x4B56_5431;
pub const VERSION: u8 = 1;
pub const HEADER_BYTES: usize = 16;

/// Caps on what a peer can make us allocate. A 4-byte length field otherwise
/// lets one bad frame ask for 4 GiB.
pub const MAX_PAYLOAD: usize = 64 << 20;
pub const MAX_BLOCKS_PER_REQUEST: usize = 256;

pub const HASH_BYTES: usize = 16;

/// How many blocks fit in one frame at a given block size.
///
/// The block cap alone is not enough: 256 blocks is 512 KiB of tiny blocks
/// but 512 MiB of Llama-3-8B blocks. Sized against `PutBlocks`, the heaviest
/// message, so the answer is safe for every opcode.
pub fn blocks_per_frame(block_bytes: usize) -> usize {
    /// Parent flag, parent hash, block count.
    const PRELUDE: usize = 1 + HASH_BYTES + 4;
    /// Per block: its name and its depth.
    const PER_BLOCK: usize = HASH_BYTES + 4;

    ((MAX_PAYLOAD - PRELUDE) / (block_bytes + PER_BLOCK)).clamp(1, MAX_BLOCKS_PER_REQUEST)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    /// Model id and block layout. A client must agree with these before its
    /// block names mean anything here.
    Info = 1,
    /// Hashes in, count of resident leading blocks out.
    MatchPrefix = 2,
    /// Hashes in, bytes for the resident leading run out.
    GetBlocks = 3,
    /// Names and bytes in, admit counts out.
    PutBlocks = 4,
    Stats = 5,
    Error = 0x7F,
}

impl Opcode {
    pub fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            1 => Opcode::Info,
            2 => Opcode::MatchPrefix,
            3 => Opcode::GetBlocks,
            4 => Opcode::PutBlocks,
            5 => Opcode::Stats,
            0x7F => Opcode::Error,
            _ => return None,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ProtoError {
    BadMagic(u32),
    BadVersion(u8),
    BadOpcode(u8),
    PayloadTooLarge(u32),
    TooManyBlocks(usize),
    /// Payload ended mid-field.
    Truncated,
    /// Payload had bytes left over, which means we disagree with the peer
    /// about the message's shape.
    TrailingBytes(usize),
    /// Peer's layout does not match ours, so its block names are meaningless.
    LayoutMismatch,
}

impl std::fmt::Display for ProtoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtoError::BadMagic(got) => write!(f, "bad magic {got:#010x}"),
            ProtoError::BadVersion(got) => write!(f, "unsupported version {got}"),
            ProtoError::BadOpcode(got) => write!(f, "unknown opcode {got}"),
            ProtoError::PayloadTooLarge(len) => write!(f, "payload {len} over cap {MAX_PAYLOAD}"),
            ProtoError::TooManyBlocks(n) => {
                write!(f, "{n} blocks over cap {MAX_BLOCKS_PER_REQUEST}")
            }
            ProtoError::Truncated => write!(f, "payload ended mid-field"),
            ProtoError::TrailingBytes(n) => write!(f, "{n} unread bytes after message"),
            ProtoError::LayoutMismatch => write!(f, "peer block layout does not match"),
        }
    }
}

impl std::error::Error for ProtoError {}

impl From<ProtoError> for std::io::Error {
    fn from(error: ProtoError) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub opcode: Opcode,
    pub flags: u8,
    /// Echoed in the response, so a client can have many requests in flight.
    pub request_id: u32,
    pub payload_len: u32,
}

impl Header {
    pub fn new(opcode: Opcode, request_id: u32, payload_len: usize) -> Self {
        Self {
            opcode,
            flags: 0,
            request_id,
            payload_len: payload_len as u32,
        }
    }

    pub fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        out[0..4].copy_from_slice(&MAGIC.to_be_bytes());
        out[4] = VERSION;
        out[5] = self.opcode as u8;
        out[6] = self.flags;
        out[8..12].copy_from_slice(&self.request_id.to_be_bytes());
        out[12..16].copy_from_slice(&self.payload_len.to_be_bytes());
        out
    }

    pub fn decode(bytes: &[u8; HEADER_BYTES]) -> Result<Self, ProtoError> {
        let magic = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(ProtoError::BadMagic(magic));
        }
        if bytes[4] != VERSION {
            return Err(ProtoError::BadVersion(bytes[4]));
        }
        let opcode = Opcode::from_u8(bytes[5]).ok_or(ProtoError::BadOpcode(bytes[5]))?;
        let payload_len = u32::from_be_bytes(bytes[12..16].try_into().unwrap());
        if payload_len as usize > MAX_PAYLOAD {
            return Err(ProtoError::PayloadTooLarge(payload_len));
        }

        Ok(Self {
            opcode,
            flags: bytes[6],
            request_id: u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
            payload_len,
        })
    }
}

/// What the server is: which model and block shape its names refer to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerInfo {
    pub model_id: String,
    pub layout: BlockLayout,
}

/// Sequential reader over a payload. Every read is bounds-checked, so a
/// malformed frame returns `Truncated` instead of panicking the server.
pub struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], ProtoError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(ProtoError::Truncated)?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or(ProtoError::Truncated)?;
        self.offset = end;
        Ok(slice)
    }

    pub fn u8(&mut self) -> Result<u8, ProtoError> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, ProtoError> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32, ProtoError> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64, ProtoError> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn hash(&mut self) -> Result<BlockHash, ProtoError> {
        let bytes: [u8; HASH_BYTES] = self.take(HASH_BYTES)?.try_into().unwrap();
        Ok(BlockHash::from_bytes(bytes))
    }

    pub fn bytes(&mut self, count: usize) -> Result<&'a [u8], ProtoError> {
        self.take(count)
    }

    pub fn string(&mut self) -> Result<String, ProtoError> {
        let len = self.u16()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| ProtoError::Truncated)
    }

    /// A block count, rejected if it exceeds the per-request cap.
    pub fn block_count(&mut self) -> Result<usize, ProtoError> {
        let count = self.u32()? as usize;
        if count > MAX_BLOCKS_PER_REQUEST {
            return Err(ProtoError::TooManyBlocks(count));
        }
        Ok(count)
    }

    pub fn hashes(&mut self) -> Result<Vec<BlockHash>, ProtoError> {
        let count = self.block_count()?;
        (0..count).map(|_| self.hash()).collect()
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    /// Assert the message consumed its whole payload.
    pub fn finish(self) -> Result<(), ProtoError> {
        match self.remaining() {
            0 => Ok(()),
            n => Err(ProtoError::TrailingBytes(n)),
        }
    }
}

#[derive(Default)]
pub struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u8(&mut self, value: u8) -> &mut Self {
        self.bytes.push(value);
        self
    }

    pub fn u16(&mut self, value: u16) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_be_bytes());
        self
    }

    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_be_bytes());
        self
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_be_bytes());
        self
    }

    pub fn hash(&mut self, hash: BlockHash) -> &mut Self {
        self.bytes.extend_from_slice(hash.as_bytes());
        self
    }

    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.bytes.extend_from_slice(bytes);
        self
    }

    pub fn string(&mut self, text: &str) -> &mut Self {
        self.u16(text.len() as u16);
        self.bytes(text.as_bytes())
    }

    pub fn hashes(&mut self, hashes: &[BlockHash]) -> &mut Self {
        self.u32(hashes.len() as u32);
        for &hash in hashes {
            self.hash(hash);
        }
        self
    }

    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

pub fn encode_info(info: &ServerInfo) -> Vec<u8> {
    let mut writer = Writer::new();
    writer
        .string(&info.model_id)
        .u32(info.layout.tokens_per_block as u32)
        .u32(info.layout.num_layers as u32)
        .u32(info.layout.num_kv_heads as u32)
        .u32(info.layout.head_dim as u32)
        .u8(dtype_code(info.layout.dtype));
    writer.finish()
}

pub fn decode_info(payload: &[u8]) -> Result<ServerInfo, ProtoError> {
    let mut reader = Reader::new(payload);
    let model_id = reader.string()?;
    let layout = BlockLayout {
        tokens_per_block: reader.u32()? as usize,
        num_layers: reader.u32()? as usize,
        num_kv_heads: reader.u32()? as usize,
        head_dim: reader.u32()? as usize,
        dtype: dtype_from_code(reader.u8()?).ok_or(ProtoError::LayoutMismatch)?,
    };
    reader.finish()?;
    Ok(ServerInfo { model_id, layout })
}

pub fn dtype_code(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::F8 => 3,
    }
}

pub fn dtype_from_code(code: u8) -> Option<DType> {
    Some(match code {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F8,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips() {
        let header = Header::new(Opcode::GetBlocks, 0xDEAD_BEEF, 4096);
        let decoded = Header::decode(&header.encode()).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn header_rejects_a_desynced_stream() {
        let mut bytes = Header::new(Opcode::Stats, 1, 0).encode();
        bytes[0] ^= 0xFF;
        assert!(matches!(
            Header::decode(&bytes),
            Err(ProtoError::BadMagic(_))
        ));
    }

    #[test]
    fn header_rejects_unknown_version_and_opcode() {
        let mut bytes = Header::new(Opcode::Stats, 1, 0).encode();
        bytes[4] = 99;
        assert_eq!(Header::decode(&bytes), Err(ProtoError::BadVersion(99)));

        let mut bytes = Header::new(Opcode::Stats, 1, 0).encode();
        bytes[5] = 42;
        assert_eq!(Header::decode(&bytes), Err(ProtoError::BadOpcode(42)));
    }

    #[test]
    fn header_refuses_an_oversized_payload() {
        let mut bytes = Header::new(Opcode::PutBlocks, 1, 0).encode();
        bytes[12..16].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(
            Header::decode(&bytes),
            Err(ProtoError::PayloadTooLarge(_))
        ));
    }

    #[test]
    fn reader_refuses_to_read_past_the_end() {
        let mut reader = Reader::new(&[0, 0, 0]);
        assert_eq!(reader.u32(), Err(ProtoError::Truncated));
    }

    #[test]
    fn reader_caps_the_block_count() {
        let mut writer = Writer::new();
        writer.u32(1_000_000);
        let payload = writer.finish();
        assert!(matches!(
            Reader::new(&payload).block_count(),
            Err(ProtoError::TooManyBlocks(1_000_000))
        ));
    }

    #[test]
    fn reader_notices_leftover_bytes() {
        let mut writer = Writer::new();
        writer.u32(7).u32(9);
        let payload = writer.finish();
        let mut reader = Reader::new(&payload);
        reader.u32().unwrap();
        assert_eq!(reader.finish(), Err(ProtoError::TrailingBytes(4)));
    }

    #[test]
    fn hashes_round_trip() {
        let hashes: Vec<BlockHash> = (0u8..4)
            .map(|i| BlockHash::from_bytes([i; HASH_BYTES]))
            .collect();
        let mut writer = Writer::new();
        writer.hashes(&hashes);
        let payload = writer.finish();

        let mut reader = Reader::new(&payload);
        assert_eq!(reader.hashes().unwrap(), hashes);
        reader.finish().unwrap();
    }

    #[test]
    fn frame_capacity_follows_block_size() {
        // Tiny blocks are limited by the count cap, real ones by bytes.
        assert_eq!(
            blocks_per_frame(BlockLayout::tiny().block_bytes()),
            MAX_BLOCKS_PER_REQUEST
        );

        let big = BlockLayout::llama3_8b().block_bytes();
        let fits = blocks_per_frame(big);
        assert!((1..MAX_BLOCKS_PER_REQUEST).contains(&fits));
        assert!(
            4 + fits * big <= MAX_PAYLOAD,
            "{fits} blocks of {big} bytes must fit a frame"
        );
        assert!(
            4 + (fits + 1) * big > MAX_PAYLOAD,
            "must not be pessimistic"
        );
    }

    #[test]
    fn info_round_trips() {
        let info = ServerInfo {
            model_id: "llama-3-8b".to_string(),
            layout: BlockLayout::llama3_8b(),
        };
        assert_eq!(decode_info(&encode_info(&info)).unwrap(), info);
    }
}
