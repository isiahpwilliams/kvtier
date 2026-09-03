//! Load generator. The number to beat is the rate the GPU regenerates KV --
//! roughly 0.6-3 GB/s for Llama-3-8B, below which recompute is cheaper.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use kvtier::block::{BlockHash, BlockLayout};
use kvtier::client::KvClient;
use kvtier::server::Server;
use kvtier::store::KvStore;
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
