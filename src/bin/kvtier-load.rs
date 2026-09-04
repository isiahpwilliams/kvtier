//! Load generator. The number to beat is the rate the GPU regenerates KV --
//! roughly 0.6-3 GB/s for Llama-3-8B, below which recompute is cheaper.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use kvtier::block::{BlockHash, BlockLayout};
use kvtier::client::KvClient;
use kvtier::server::{Server, WritebackConfig};
use kvtier::store::KvStore;
use kvtier::tier::DiskTier;
use kvtier::trace::SplitMix64;
use tokio::sync::Mutex;

const BLOCKS_PER_REQUEST: [usize; 4] = [1, 8, 32, 128];
const ROUNDS: usize = 60;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let layout = BlockLayout::llama3_8b();
    let block_bytes = layout.block_bytes();
    let total_blocks = 256;

    let store = KvStore::new("bench", layout.clone(), total_blocks + 8)?;
    let server = Server::bind("127.0.0.1:0".parse().unwrap(), Arc::new(Mutex::new(store))).await?;
    let addr: SocketAddr = server.local_addr()?;
    tokio::spawn(server.run());

    let mut client = KvClient::connect(addr).await?;
    println!(
        "block {} MiB, {} blocks resident\n",
        block_bytes / (1 << 20),
        total_blocks
    );

    // Fill the server with a single chain so any prefix of it is a hit.
    let mut rng = SplitMix64::new(7);
    let hashes: Vec<BlockHash> = (0..total_blocks)
        .map(|_| {
            let mut bytes = [0u8; 16];
            bytes[..8].copy_from_slice(&rng.next_u64().to_le_bytes());
            bytes[8..].copy_from_slice(&rng.next_u64().to_le_bytes());
            BlockHash::from_bytes(bytes)
        })
        .collect();

    let payload = vec![0xABu8; block_bytes];
    let mut parent = None;
    for (i, &hash) in hashes.iter().enumerate() {
        let names = [(hash, ((i + 1) * layout.tokens_per_block) as u32)];
        client.put_blocks(parent, &names, &[&payload]).await?;
        parent = Some(hash);
    }

    println!(
        "{:>8}  {:>10}  {:>12}  {:>12}",
        "blocks", "MiB/req", "GB/s", "us/req"
    );
    for count in BLOCKS_PER_REQUEST {
        let request = &hashes[..count];
        let bytes_per_request = count * block_bytes;

        // Warm the path before timing it.
        client.get_blocks(request).await?;

        let start = Instant::now();
        for _ in 0..ROUNDS {
            let blocks = client.get_blocks(request).await?;
            assert_eq!(blocks.len(), count);
        }
        let elapsed = start.elapsed();

        let total = (bytes_per_request * ROUNDS) as f64;
        println!(
            "{count:>8}  {:>10.1}  {:>12.2}  {:>12.0}",
            bytes_per_request as f64 / (1 << 20) as f64,
            total / elapsed.as_secs_f64() / 1e9,
            elapsed.as_micros() as f64 / ROUNDS as f64
        );
    }

    // A MatchPrefix carries no KV, so this is the pure round-trip floor --
    // what a miss costs a request that gets nothing back.
    let start = Instant::now();
    for _ in 0..1000 {
        client.match_prefix(&hashes[..32]).await?;
    }
    println!(
        "\nmatch_prefix round trip: {:.1} us",
        start.elapsed().as_micros() as f64 / 1000.0
    );

    concurrency_sweep(addr, &hashes[..8], block_bytes).await?;
    cold_fetch_sweep(&layout).await?;
    writeback_sweep(&layout).await?;
    Ok(())
}

/// Admissions under memory pressure, with and without background writeback.
///
/// Every admit past capacity has to push something out of RAM. Without
/// writeback that means a 2 MiB write with the store lock held, on the
/// critical path of a request. With it, the block already has a copy on disk
/// and the eviction is bookkeeping.
async fn writeback_sweep(layout: &BlockLayout) -> std::io::Result<()> {
    const RAM: usize = 64;
    const BLOCKS: usize = 512;

    let block_bytes = layout.block_bytes();
    println!("\nadmissions under pressure: {BLOCKS} blocks into {RAM} of RAM");
    println!(
        "{:>12}  {:>12}  {:>12}  {:>12}",
        "writeback", "ms total", "blocking", "written back"
    );

    for enabled in [false, true] {
        let disk = DiskTier::temporary(block_bytes, BLOCKS * 2)?;
        disk.try_bypass_page_cache();
        let store = KvStore::new("bench", layout.clone(), RAM)?.with_disk_tier(disk);
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), Arc::new(Mutex::new(store)))
            .await?
            .with_writeback(enabled.then(|| WritebackConfig {
                interval: Duration::from_millis(1),
                batch: 32,
                watermark: 0.5,
            }));
        let addr: SocketAddr = server.local_addr()?;
        tokio::spawn(server.run());

        let mut client = KvClient::connect(addr).await?;
        let payload = vec![0xEFu8; block_bytes];
        let mut rng = SplitMix64::new(enabled as u64);

        let start = Instant::now();
        for depth in 0..BLOCKS {
            let mut bytes = [0u8; 16];
            bytes[..8].copy_from_slice(&rng.next_u64().to_le_bytes());
            bytes[8..].copy_from_slice(&rng.next_u64().to_le_bytes());
            let hash = BlockHash::from_bytes(bytes);

            // Independent blocks, so every one past capacity forces an eviction.
            let names = [(hash, ((depth + 1) * layout.tokens_per_block) as u32)];
            client.put_blocks(None, &names, &[&payload]).await?;
        }
        let elapsed = start.elapsed();

        let (stats, _) = client.stats().await?;
        println!(
            "{:>12}  {:>12.0}  {:>12}  {:>12}",
            if enabled { "on" } else { "off" },
            elapsed.as_secs_f64() * 1000.0,
            stats.blocking_demotions,
            stats.written_back
        );
    }
    Ok(())
}

/// Cold fetches, where most of the run has to come off disk.
///
/// `MatchPrefix` names the whole run before any of it is touched, so the
/// reads need no prediction -- the only question is how many to have in
/// flight. Note this reads a file written moments ago, so the page cache is
/// warm: these numbers bound our own overhead, not the drive's latency.
async fn cold_fetch_sweep(layout: &BlockLayout) -> std::io::Result<()> {
    const CHAINS: usize = 8;
    const PER_CHAIN: usize = 31; // one frame at this block size
    const ROUNDS: usize = 4;

    let block_bytes = layout.block_bytes();
    println!("\ncold fetch: {CHAINS} chains x {PER_CHAIN} blocks, 64 blocks of RAM");
    println!(
        "{:>10}  {:>10}  {:>10}  {:>12}",
        "in flight", "ms/fetch", "GB/s", "2 clients"
    );

    for parallel in [1usize, 4, 8] {
        let disk = DiskTier::temporary(block_bytes, CHAINS * PER_CHAIN + 16)?;
        let store = KvStore::new("bench", layout.clone(), 64)?.with_disk_tier(disk);
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), Arc::new(Mutex::new(store)))
            .await?
            .with_parallel_reads(parallel);
        let addr: SocketAddr = server.local_addr()?;
        tokio::spawn(server.run());

        let mut client = KvClient::connect(addr).await?;
        let payload = vec![0xCDu8; block_bytes];

        // Independent chains, so fetching one pushes the others to disk.
        let mut chains = Vec::new();
        let mut rng = SplitMix64::new(parallel as u64);
        for _ in 0..CHAINS {
            let mut hashes = Vec::new();
            let mut parent = None;
            for depth in 0..PER_CHAIN {
                let mut bytes = [0u8; 16];
                bytes[..8].copy_from_slice(&rng.next_u64().to_le_bytes());
                bytes[8..].copy_from_slice(&rng.next_u64().to_le_bytes());
                let hash = BlockHash::from_bytes(bytes);

                let names = [(hash, ((depth + 1) * layout.tokens_per_block) as u32)];
                client.put_blocks(parent, &names, &[&payload]).await?;
                parent = Some(hash);
                hashes.push(hash);
            }
            chains.push(hashes);
        }

        // One client at a time: does queue depth help the drive?
        let start = Instant::now();
        for _ in 0..ROUNDS {
            for chain in &chains {
                let blocks = client.get_blocks(chain).await?;
                assert_eq!(blocks.len(), PER_CHAIN);
            }
        }
        let elapsed = start.elapsed();

        // Several clients at once: is the store lock still held across the
        // read? If it were, this would not move. Kept to two clients because
        // a fetch reserves a RAM slot per demoted block, and more than that
        // would not fit -- the runs would come back truncated and the number
        // would be measuring bytes that never moved.
        let concurrent = Instant::now();
        let mut tasks = Vec::new();
        for chain in chains.iter().take(2) {
            let chain = chain.clone();
            tasks.push(tokio::spawn(async move {
                let mut client = KvClient::connect(addr).await.unwrap();
                let mut blocks = 0usize;
                for _ in 0..ROUNDS {
                    blocks += client.get_blocks(&chain).await.unwrap().len();
                }
                blocks
            }));
        }
        let mut moved = 0usize;
        for task in tasks {
            moved += task.await.unwrap();
        }
        let concurrent = concurrent.elapsed();

        let fetches = (ROUNDS * CHAINS) as f64;
        let total = fetches * (PER_CHAIN * block_bytes) as f64;
        println!(
            "{parallel:>10}  {:>10.1}  {:>10.2}  {:>12.2}",
            elapsed.as_secs_f64() * 1000.0 / fetches,
            total / elapsed.as_secs_f64() / 1e9,
            (moved * block_bytes) as f64 / concurrent.as_secs_f64() / 1e9
        );
    }
    Ok(())
}

/// Aggregate throughput as clients pile on. Flat means they are serializing
/// on something; the store lock held across the socket write is the suspect.
async fn concurrency_sweep(
    addr: SocketAddr,
    request: &[BlockHash],
    block_bytes: usize,
) -> std::io::Result<()> {
    const ROUNDS: usize = 50;

    println!("\n{:>8}  {:>12}  {:>12}", "clients", "GB/s", "ms/client");
    for clients in [1usize, 2, 4, 8] {
        let mut tasks = Vec::new();
        for _ in 0..clients {
            let request = request.to_vec();
            tasks.push(tokio::spawn(async move {
                let mut client = KvClient::connect(addr).await.unwrap();
                // Connect outside the timed region as far as we can: the
                // barrier below is what actually lines the clients up.
                client.get_blocks(&request).await.unwrap();
                client
            }));
        }
        let mut ready = Vec::new();
        for task in tasks {
            ready.push(task.await.unwrap());
        }

        let start = Instant::now();
        let mut running = Vec::new();
        for mut client in ready {
            let request = request.to_vec();
            running.push(tokio::spawn(async move {
                for _ in 0..ROUNDS {
                    client.get_blocks(&request).await.unwrap();
                }
            }));
        }
        for task in running {
            task.await.unwrap();
        }
        let elapsed = start.elapsed();

        let total = (clients * ROUNDS * request.len() * block_bytes) as f64;
        println!(
            "{clients:>8}  {:>12.2}  {:>12.1}",
            total / elapsed.as_secs_f64() / 1e9,
            elapsed.as_millis() as f64
        );
    }
    Ok(())
}
