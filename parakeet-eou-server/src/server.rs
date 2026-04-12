use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use serde_json::json;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info, warn};

use crate::audio::{self, ClientMessage};
use crate::metrics::Metrics;
use crate::transcriber::{Event, Transcriber};

#[derive(Clone)]
pub struct AppState {
    pub model_dir: PathBuf,
    pub max_sessions: usize,
    pub active_sessions: Arc<AtomicUsize>,
    pub metrics: Arc<Metrics>,
    pub device: &'static str,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/asr", get(ws_handler))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let body = json!({
        "status": "ok",
        "model": "parakeet-eou-120m",
        "device": state.device,
        "sessions": state.active_sessions.load(Ordering::Relaxed),
        "max_sessions": state.max_sessions,
        "avg_latency_ms": state.metrics.avg_latency_ms(),
        "model_load_time_ms": state.metrics.model_load_time_ms,
    });
    (StatusCode::OK, axum::Json(body))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let current = state.active_sessions.load(Ordering::Relaxed);
    if current >= state.max_sessions {
        warn!(
            "rejecting WebSocket: {} active sessions (max {})",
            current, state.max_sessions
        );
        return (StatusCode::SERVICE_UNAVAILABLE, "max sessions reached").into_response();
    }
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let session_id = state.active_sessions.fetch_add(1, Ordering::Relaxed) + 1;
    let started = Instant::now();
    info!(session = session_id, "websocket connected");

    let transcriber = match Transcriber::new(&state.model_dir, state.metrics.clone()) {
        Ok(t) => t,
        Err(e) => {
            error!(session = session_id, error = %e, "failed to init transcriber");
            let _ = socket
                .send(Message::Text(
                    serde_json::to_string(&Event::Error {
                        message: format!("model init failed: {e}"),
                    })
                    .unwrap()
                    .into(),
                ))
                .await;
            state.active_sessions.fetch_sub(1, Ordering::Relaxed);
            return;
        }
    };

    let total_samples = run_session(&mut socket, transcriber, session_id).await;

    let prev = state.active_sessions.fetch_sub(1, Ordering::Relaxed);
    let audio_seconds = total_samples as f64 / 16_000.0;
    info!(
        session = session_id,
        duration_s = started.elapsed().as_secs_f64(),
        audio_s = audio_seconds,
        remaining = prev - 1,
        "websocket closed"
    );
}

async fn run_session(
    socket: &mut WebSocket,
    mut transcriber: Transcriber,
    session_id: usize,
) -> usize {
    let mut total_samples = 0usize;

    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                warn!(session = session_id, error = %e, "recv error");
                break;
            }
        };

        match msg {
            Message::Binary(bytes) => {
                match audio::decode_binary_f32(&bytes) {
                    Ok(samples) => {
                        total_samples += samples.len();
                        let events = transcriber.push(&samples);
                        if !send_events(socket, &events).await {
                            return total_samples;
                        }
                    }
                    Err(e) => {
                        let _ = send_error(socket, e.to_string()).await;
                    }
                }
            }
            Message::Text(txt) => {
                let parsed: Result<ClientMessage, _> = serde_json::from_str(&txt);
                match parsed {
                    Ok(ClientMessage::Audio {
                        data,
                        sample_rate,
                        encoding,
                    }) => match audio::decode_json_audio(&data, sample_rate, encoding.as_deref()) {
                        Ok(samples) => {
                            total_samples += samples.len();
                            let events = transcriber.push(&samples);
                            if !send_events(socket, &events).await {
                                return total_samples;
                            }
                        }
                        Err(e) => {
                            let _ = send_error(socket, e.to_string()).await;
                        }
                    },
                    Ok(ClientMessage::Eos) => {
                        debug!(session = session_id, "client sent eos");
                        let events = transcriber.flush();
                        let _ = send_events(socket, &events).await;
                    }
                    Err(e) => {
                        let _ = send_error(socket, format!("invalid json message: {e}")).await;
                    }
                }
            }
            Message::Close(_) => {
                debug!(session = session_id, "client sent close");
                let events = transcriber.flush();
                let _ = send_events(socket, &events).await;
                break;
            }
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }

    total_samples
}

async fn send_events(socket: &mut WebSocket, events: &[Event]) -> bool {
    for ev in events {
        let json = match serde_json::to_string(ev) {
            Ok(j) => j,
            Err(e) => {
                error!(error = %e, "failed to serialize event");
                continue;
            }
        };
        if let Err(e) = socket.send(Message::Text(json.into())).await {
            warn!(error = %e, "send failed; closing session");
            return false;
        }
    }
    true
}

async fn send_error(socket: &mut WebSocket, message: String) -> bool {
    let ev = Event::Error { message };
    send_events(socket, std::slice::from_ref(&ev)).await
}
