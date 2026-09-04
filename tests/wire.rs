//! Client and server over a real socket.

use std::net::SocketAddr;
use std::sync::Arc;

use kvtier::block::{BlockHash, BlockLayout, TokenId};
use kvtier::client::KvClient;
use kvtier::proto::{HEADER_BYTES, Header, MAGIC, Opcode};
use kvtier::server::Server;
use kvtier::store::KvStore;
use kvtier::trace::{self, SplitMix64};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

/// A server on an ephemeral port, plus a handle to its store so a test can
/// check what actually landed.
async fn start(capacity: usize) -> (SocketAddr, Arc<Mutex<KvStore>>) {
    let store = Arc::new(Mutex::new(
        KvStore::new("test-model", BlockLayout::tiny(), capacity).unwrap(),
    ));
    let server = Server::bind("127.0.0.1:0".parse().unwrap(), Arc::clone(&store))
        .await
        .unwrap();
    let addr = server.local_addr().unwrap();
    tokio::spawn(server.run());
    (addr, store)
}

fn tokens(count: usize, seed: u64) -> Vec<TokenId> {
    SplitMix64::new(seed).tokens(count, 32_000)
}

/// Name a sequence and build its payloads, without touching the server.
fn sequence(tokens: &[TokenId]) -> (Vec<(BlockHash, u32)>, Vec<Vec<u8>>) {
    let layout = BlockLayout::tiny();
    let hasher = kvtier::block::PrefixHasher::new("test-model", &layout);
    let hashes = hasher.chain(tokens);

    let names = hashes
        .iter()
        .enumerate()
        .map(|(i, &hash)| (hash, ((i + 1) * layout.tokens_per_block) as u32))
        .collect();
    let payloads = hashes
        .iter()
        .map(|&hash| trace::block_payload(hash, layout.block_bytes()))
        .collect();
    (names, payloads)
}

#[tokio::test]
async fn info_reports_the_servers_layout() {
    let (addr, _store) = start(64).await;
    let client = KvClient::connect(addr).await.unwrap();

    assert_eq!(client.info().model_id, "test-model");
    assert_eq!(client.info().layout, BlockLayout::tiny());
    assert_eq!(client.block_bytes(), BlockLayout::tiny().block_bytes());
}

#[tokio::test]
async fn a_mismatched_layout_is_refused_at_connect() {
    let (addr, _store) = start(64).await;
    // Names computed under llama-3-8b's shape mean nothing to a tiny server.
    let result = KvClient::connect_expecting(addr, &BlockLayout::llama3_8b()).await;
    let error = result.err().expect("must refuse a mismatched layout");
    assert!(error.to_string().contains("layout"), "got {error}");
}

#[tokio::test]
async fn put_then_get_returns_the_same_bytes() {
    let (addr, _store) = start(64).await;
    let mut client = KvClient::connect(addr).await.unwrap();

    let tokens = tokens(64, 1);
    let (names, payloads) = sequence(&tokens);
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();

    let report = client.put_blocks(None, &names, &refs).await.unwrap();
    assert_eq!(report.inserted, 4);

    let hashes: Vec<BlockHash> = names.iter().map(|&(hash, _)| hash).collect();
    assert_eq!(client.match_prefix(&hashes).await.unwrap(), 4);

    let fetched = client.get_blocks(&hashes).await.unwrap();
    assert_eq!(fetched, payloads, "bytes must survive the round trip");
}

#[tokio::test]
async fn a_cold_server_returns_nothing_rather_than_failing() {
    let (addr, _store) = start(64).await;
    let mut client = KvClient::connect(addr).await.unwrap();

    let (names, _) = sequence(&tokens(64, 2));
    let hashes: Vec<BlockHash> = names.iter().map(|&(hash, _)| hash).collect();

    assert_eq!(client.match_prefix(&hashes).await.unwrap(), 0);
    assert!(client.get_blocks(&hashes).await.unwrap().is_empty());
}

#[tokio::test]
async fn get_returns_the_leading_run_and_stops_at_a_gap() {
    let (addr, _store) = start(64).await;
    let mut client = KvClient::connect(addr).await.unwrap();

    let tokens = tokens(96, 3);
    let (names, payloads) = sequence(&tokens);
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();

    // Store only the first three of six blocks.
    client
        .put_blocks(None, &names[..3], &refs[..3])
        .await
        .unwrap();

    let hashes: Vec<BlockHash> = names.iter().map(|&(hash, _)| hash).collect();
    let fetched = client.get_blocks(&hashes).await.unwrap();
    assert_eq!(fetched.len(), 3, "must not send bytes past the gap");
    assert_eq!(fetched, payloads[..3]);
}

#[tokio::test]
async fn a_second_client_hits_on_the_firsts_work() {
    // The whole point of the phase: two engine processes sharing a prefix.
    let (addr, _store) = start(64).await;

    let system_prompt = tokens(48, 4);
    let mut first_tokens = system_prompt.clone();
    first_tokens.extend(tokens(32, 5));
    let mut second_tokens = system_prompt.clone();
    second_tokens.extend(tokens(32, 6));

    let mut first = KvClient::connect(addr).await.unwrap();
    let (names, payloads) = sequence(&first_tokens);
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    first.put_blocks(None, &names, &refs).await.unwrap();

    let mut second = KvClient::connect(addr).await.unwrap();
    let (other_names, _) = sequence(&second_tokens);
    let hashes: Vec<BlockHash> = other_names.iter().map(|&(hash, _)| hash).collect();

    assert_eq!(
        second.match_prefix(&hashes).await.unwrap(),
        3,
        "the shared system prompt must hit across connections"
    );
    let fetched = second.get_blocks(&hashes).await.unwrap();
    assert_eq!(fetched, payloads[..3]);
}

#[tokio::test]
async fn an_orphan_put_is_rejected_over_the_wire() {
    let (addr, store) = start(64).await;
    let mut client = KvClient::connect(addr).await.unwrap();

    let (names, payloads) = sequence(&tokens(64, 7));
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();

    // Offer block 2 onward, claiming block 1 as parent, without block 1.
    let report = client
        .put_blocks(Some(names[1].0), &names[2..], &refs[2..])
        .await
        .unwrap();
    assert_eq!(report.inserted, 0);
    assert_eq!(report.dropped, 2);
    assert_eq!(store.lock().await.resident_blocks(), 0);
}

#[tokio::test]
async fn stats_come_back_from_the_server() {
    let (addr, _store) = start(64).await;
    let mut client = KvClient::connect(addr).await.unwrap();

    let (names, payloads) = sequence(&tokens(64, 8));
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    client.put_blocks(None, &names, &refs).await.unwrap();
    client.put_blocks(None, &names, &refs).await.unwrap();

    let (stats, resident) = client.stats().await.unwrap();
    assert_eq!(resident, 4);
    assert_eq!(stats.inserted_blocks, 4);
    assert_eq!(stats.deduped_blocks, 4, "the second put was all dedup");
}

#[tokio::test]
async fn many_requests_share_one_connection() {
    let (addr, _store) = start(512).await;
    let mut client = KvClient::connect(addr).await.unwrap();

    for seed in 0..32 {
        let (names, payloads) = sequence(&tokens(32, 100 + seed));
        let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
        client.put_blocks(None, &names, &refs).await.unwrap();

        let hashes: Vec<BlockHash> = names.iter().map(|&(hash, _)| hash).collect();
        assert_eq!(client.match_prefix(&hashes).await.unwrap(), 2);
    }
}

#[tokio::test]
async fn a_bad_frame_gets_an_error_and_the_connection_survives() {
    let (addr, _store) = start(64).await;
    let mut socket = TcpStream::connect(addr).await.unwrap();

    // Claim 1,000,000 hashes but send none.
    let header = Header::new(Opcode::MatchPrefix, 42, 4);
    socket.write_all(&header.encode()).await.unwrap();
    socket.write_all(&1_000_000u32.to_be_bytes()).await.unwrap();

    let mut bytes = [0u8; HEADER_BYTES];
    socket.read_exact(&mut bytes).await.unwrap();
    let response = Header::decode(&bytes).unwrap();
    assert_eq!(response.opcode, Opcode::Error);
    assert_eq!(response.request_id, 42, "errors must be attributable");

    let mut body = vec![0u8; response.payload_len as usize];
    socket.read_exact(&mut body).await.unwrap();

    // Still usable afterwards: a peer's bad frame is not our problem.
    let header = Header::new(Opcode::Stats, 43, 0);
    socket.write_all(&header.encode()).await.unwrap();
    socket.read_exact(&mut bytes).await.unwrap();
    assert_eq!(Header::decode(&bytes).unwrap().opcode, Opcode::Stats);
}

#[tokio::test]
async fn a_desynced_stream_is_cut_off() {
    let (addr, _store) = start(64).await;
    let mut socket = TcpStream::connect(addr).await.unwrap();

    let mut bytes = Header::new(Opcode::Stats, 1, 0).encode();
    bytes[0..4].copy_from_slice(&(MAGIC ^ 0xFFFF_FFFF).to_be_bytes());
    socket.write_all(&bytes).await.unwrap();

    // Bad magic means we no longer know where frames begin, so the server
    // must hang up rather than guess.
    let mut response = [0u8; HEADER_BYTES];
    assert!(socket.read_exact(&mut response).await.is_err());
}

#[tokio::test]
async fn a_run_longer_than_one_frame_is_split_transparently() {
    // Real block sizes overflow a frame long before the block-count cap:
    // 32 Llama-3-8B blocks is already 64 MiB. The client must stitch the
    // pieces back together without the caller knowing.
    let layout = BlockLayout::llama3_8b();
    let block_bytes = layout.block_bytes();
    let per_frame = kvtier::proto::blocks_per_frame(block_bytes);
    let count = per_frame + 4;

    let store = Arc::new(Mutex::new(
        KvStore::new("big", layout.clone(), count + 2).unwrap(),
    ));
    let server = Server::bind("127.0.0.1:0".parse().unwrap(), Arc::clone(&store))
        .await
        .unwrap();
    let addr = server.local_addr().unwrap();
    tokio::spawn(server.run());

    let mut client = KvClient::connect(addr).await.unwrap();
    let hasher = kvtier::block::PrefixHasher::new("big", &layout);
    let tokens: Vec<TokenId> = SplitMix64::new(11).tokens(count * layout.tokens_per_block, 32_000);
    let hashes = hasher.chain(&tokens);

    let names: Vec<(BlockHash, u32)> = hashes
        .iter()
        .enumerate()
        .map(|(i, &hash)| (hash, ((i + 1) * layout.tokens_per_block) as u32))
        .collect();
    // One distinguishable byte per block is enough to catch a mis-stitch.
    let payloads: Vec<Vec<u8>> = (0..count).map(|i| vec![i as u8; block_bytes]).collect();
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();

    let report = client.put_blocks(None, &names, &refs).await.unwrap();
    assert_eq!(report.inserted, count, "a multi-frame put must land whole");

    let fetched = client.get_blocks(&hashes).await.unwrap();
    assert_eq!(
        fetched.len(),
        count,
        "a multi-frame get must come back whole"
    );
    for (i, block) in fetched.iter().enumerate() {
        assert!(
            block.iter().all(|&b| b == i as u8),
            "block {i} is stitched wrong"
        );
    }
}

#[tokio::test]
async fn concurrent_fetches_each_get_their_own_bytes() {
    // The point of pinning: transfers overlap, and none of them sees another
    // client's block.
    let (addr, _store) = start(512).await;

    let mut writer = KvClient::connect(addr).await.unwrap();
    let mut sequences = Vec::new();
    for seed in 0..8u64 {
        let (names, payloads) = sequence(&tokens(64, 200 + seed));
        let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
        writer.put_blocks(None, &names, &refs).await.unwrap();
        sequences.push((
            names.iter().map(|&(hash, _)| hash).collect::<Vec<_>>(),
            payloads,
        ));
    }

    let mut tasks = Vec::new();
    for (hashes, payloads) in sequences {
        tasks.push(tokio::spawn(async move {
            let mut client = KvClient::connect(addr).await.unwrap();
            for _ in 0..8 {
                assert_eq!(client.get_blocks(&hashes).await.unwrap(), payloads);
            }
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
}

#[tokio::test]
async fn admits_proceed_while_a_fetch_is_in_flight() {
    // A reader holds pins, not the store lock, so writers are not blocked and
    // the reader's bytes are unaffected by what lands meanwhile.
    let (addr, store) = start(512).await;

    let mut writer = KvClient::connect(addr).await.unwrap();
    let (names, payloads) = sequence(&tokens(64, 300));
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    writer.put_blocks(None, &names, &refs).await.unwrap();
    let hashes: Vec<BlockHash> = names.iter().map(|&(hash, _)| hash).collect();

    let reader = tokio::spawn(async move {
        let mut client = KvClient::connect(addr).await.unwrap();
        for _ in 0..32 {
            assert_eq!(client.get_blocks(&hashes).await.unwrap(), payloads);
        }
    });

    for seed in 0..32u64 {
        let (names, payloads) = sequence(&tokens(32, 400 + seed));
        let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
        writer.put_blocks(None, &names, &refs).await.unwrap();
    }
    reader.await.unwrap();

    assert!(store.lock().await.resident_blocks() > 4, "admits landed");
}

#[tokio::test]
async fn pins_are_released_after_a_fetch() {
    let (addr, store) = start(64).await;
    let mut client = KvClient::connect(addr).await.unwrap();

    let (names, payloads) = sequence(&tokens(64, 500));
    let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
    client.put_blocks(None, &names, &refs).await.unwrap();

    let hashes: Vec<BlockHash> = names.iter().map(|&(hash, _)| hash).collect();
    client.get_blocks(&hashes).await.unwrap();

    // A leaked pin would leave the block unevictable forever, which shows up
    // as an empty eviction candidate list.
    let store = store.lock().await;
    assert_eq!(
        store.index().leaves().count(),
        1,
        "the tail block must be evictable again once the transfer is done"
    );
}

#[tokio::test]
async fn a_client_fetching_demoted_blocks_gets_the_right_bytes() {
    // The server reads them back off disk, in parallel, with the store lock
    // released. The client should not be able to tell.
    let layout = BlockLayout::tiny();
    let disk = kvtier::tier::DiskTier::temporary(layout.block_bytes(), 256).unwrap();
    let store = Arc::new(Mutex::new(
        KvStore::new("test-model", layout, 4)
            .unwrap()
            .with_disk_tier(disk),
    ));
    let server = Server::bind("127.0.0.1:0".parse().unwrap(), Arc::clone(&store))
        .await
        .unwrap();
    let addr = server.local_addr().unwrap();
    tokio::spawn(server.run());

    let mut client = KvClient::connect(addr).await.unwrap();
    let mut written = Vec::new();
    for seed in 0..12u64 {
        let (names, payloads) = sequence(&tokens(64, 700 + seed));
        let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
        client.put_blocks(None, &names, &refs).await.unwrap();
        written.push((
            names.iter().map(|&(hash, _)| hash).collect::<Vec<_>>(),
            payloads,
        ));
    }

    // Only 4 blocks fit in RAM, so most of this comes off disk.
    for (hashes, payloads) in &written {
        assert_eq!(client.get_blocks(hashes).await.unwrap(), *payloads);
    }

    let (stats, resident) = client.stats().await.unwrap();
    assert_eq!(resident, 48, "nothing was dropped");
    assert!(stats.promoted_blocks > 0, "blocks came back off disk");
}

#[tokio::test]
async fn the_server_cleans_blocks_in_the_background() {
    let layout = BlockLayout::tiny();
    let disk = kvtier::tier::DiskTier::temporary(layout.block_bytes(), 256).unwrap();
    let store = Arc::new(Mutex::new(
        KvStore::new("test-model", layout, 16)
            .unwrap()
            .with_disk_tier(disk),
    ));
    let server = Server::bind("127.0.0.1:0".parse().unwrap(), Arc::clone(&store))
        .await
        .unwrap()
        .with_writeback(Some(kvtier::server::WritebackConfig {
            interval: std::time::Duration::from_millis(1),
            batch: 32,
            watermark: 0.5,
        }));
    let addr = server.local_addr().unwrap();
    tokio::spawn(server.run());

    let mut client = KvClient::connect(addr).await.unwrap();
    for seed in 0..4u64 {
        let (names, payloads) = sequence(&tokens(64, 900 + seed));
        let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
        client.put_blocks(None, &names, &refs).await.unwrap();
    }

    // RAM is over the watermark, so the loop has work to do.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let (stats, _) = client.stats().await.unwrap();
    assert!(stats.written_back > 0, "the writeback loop must have run");
    assert_eq!(store.lock().await.dirty_blocks(), 0, "everything is clean");
}
