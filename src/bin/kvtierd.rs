//! The cache tier daemon.

use std::sync::Arc;

use kvtier::block::BlockLayout;
use kvtier::server::Server;
use kvtier::store::KvStore;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let addr = std::env::var("KVTIER_ADDR").unwrap_or_else(|_| "127.0.0.1:7431".to_string());
    let model_id = std::env::var("KVTIER_MODEL").unwrap_or_else(|_| "llama-3-8b".to_string());
    let capacity: usize = std::env::var("KVTIER_BLOCKS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4096);

    let layout = BlockLayout::llama3_8b();
    let store = KvStore::new(&model_id, layout.clone(), capacity)?;

    println!(
        "kvtier: {model_id}, {} MiB blocks x {capacity} = {} GiB reserved",
        layout.block_bytes() / (1 << 20),
        (layout.block_bytes() * capacity) as f64 / (1u64 << 30) as f64
    );

    let server = Server::bind(
        addr.parse().expect("KVTIER_ADDR"),
        Arc::new(Mutex::new(store)),
    )
    .await?;
    println!("listening on {}", server.local_addr()?);
    server.run().await
}
