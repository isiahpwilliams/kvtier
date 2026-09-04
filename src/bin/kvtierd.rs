//! The cache tier daemon.

use std::sync::Arc;

use kvtier::block::BlockLayout;
use kvtier::server::Server;
use kvtier::store::KvStore;
use kvtier::tier::DiskTier;
use tokio::sync::Mutex;

fn env_or<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let addr: String = env_or("KVTIER_ADDR", "127.0.0.1:7431".to_string());
    let model_id: String = env_or("KVTIER_MODEL", "llama-3-8b".to_string());
    let capacity: usize = env_or("KVTIER_BLOCKS", 4096);
    let disk_blocks: usize = env_or("KVTIER_DISK_BLOCKS", 0);

    let layout = match env_or("KVTIER_LAYOUT", "llama3-8b".to_string()).as_str() {
        "tiny" => BlockLayout::tiny(),
        "llama3-8b" => BlockLayout::llama3_8b(),
        other => {
            eprintln!("unknown KVTIER_LAYOUT {other:?}; expected tiny or llama3-8b");
            std::process::exit(2);
        }
    };

    let mut store = KvStore::new(&model_id, layout.clone(), capacity)?;
    if disk_blocks > 0 {
        store = store.with_disk_tier(DiskTier::temporary(layout.block_bytes(), disk_blocks)?);
    }

    println!(
        "kvtier: {model_id}, {} KiB blocks x {capacity} in RAM, {disk_blocks} on disk",
        layout.block_bytes() / 1024,
    );

    let server = Server::bind(
        addr.parse().expect("KVTIER_ADDR"),
        Arc::new(Mutex::new(store)),
    )
    .await?;
    println!("listening on {}", server.local_addr()?);
    server.run().await
}
