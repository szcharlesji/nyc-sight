mod audio;
mod metrics;
mod server;
mod transcriber;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use parakeet_rs::ParakeetEOU;
use tokio::net::TcpListener;
use tokio::signal;
use tracing::{error, info};
use tracing_subscriber::{filter::EnvFilter, fmt, prelude::*};

use crate::metrics::Metrics;
use crate::server::{build_router, AppState};

#[derive(Parser, Debug)]
#[command(name = "parakeet-eou-server", version, about = "Streaming Parakeet EOU ASR over WebSocket")]
struct Args {
    /// Directory containing encoder.onnx, decoder_joint.onnx, tokenizer.json
    #[arg(long, default_value = "./models/eou")]
    model_dir: PathBuf,

    /// Bind host (default: 0.0.0.0 — reachable over Tailscale)
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Bind port
    #[arg(long, default_value_t = 3030)]
    port: u16,

    /// Maximum concurrent WebSocket sessions
    #[arg(long, default_value_t = 2)]
    max_sessions: usize,

    /// Log level (overridden by RUST_LOG)
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing(&args.log_level);

    let device = if cfg!(feature = "cuda") { "cuda" } else { "cpu" };

    info!(
        model_dir = %args.model_dir.display(),
        host = %args.host,
        port = args.port,
        max_sessions = args.max_sessions,
        device,
        "starting parakeet-eou-server"
    );

    validate_model_dir(&args.model_dir)?;

    // Load-once sanity check: catches missing files / bad ONNX at startup instead of on first WS.
    let load_start = Instant::now();
    let _probe = ParakeetEOU::from_pretrained(&args.model_dir, None)
        .with_context(|| format!("failed to load model from {}", args.model_dir.display()))?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    drop(_probe);
    info!(load_time_ms = load_ms, "model probe loaded ok");

    let state = AppState {
        model_dir: args.model_dir.clone(),
        max_sessions: args.max_sessions,
        active_sessions: Arc::new(AtomicUsize::new(0)),
        metrics: Arc::new(Metrics::new(load_ms)),
        device,
    };

    let app = build_router(state.clone());

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .with_context(|| format!("invalid bind address {}:{}", args.host, args.port))?;
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    info!(%addr, "listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state.clone()))
        .await?;

    info!(
        sessions_at_exit = state
            .active_sessions
            .load(std::sync::atomic::Ordering::Relaxed),
        "shutdown complete"
    );
    Ok(())
}

fn init_tracing(level: &str) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level));
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(false))
        .init();
}

fn validate_model_dir(dir: &PathBuf) -> Result<()> {
    let required = ["encoder.onnx", "decoder_joint.onnx", "tokenizer.json"];
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|f| !dir.join(f).exists())
        .collect();
    if missing.is_empty() {
        return Ok(());
    }

    let listing: Vec<String> = std::fs::read_dir(dir)
        .map(|it| {
            it.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_else(|_| vec!["<directory unreadable>".into()]);

    error!(
        "model directory {} is missing: {:?}. contents: {:?}",
        dir.display(),
        missing,
        listing
    );
    anyhow::bail!(
        "missing model files in {}: {:?}. Run scripts/download_model.sh.",
        dir.display(),
        missing
    );
}

async fn shutdown_signal(state: AppState) {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("received Ctrl-C, draining sessions"),
        _ = terminate => info!("received SIGTERM, draining sessions"),
    }

    info!(
        active = state
            .active_sessions
            .load(std::sync::atomic::Ordering::Relaxed),
        "shutting down"
    );
}
