use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use parakeet_rs::ParakeetEOU;
use serde::Serialize;

use crate::metrics::Metrics;

pub const CHUNK_SIZE: usize = 2560; // 160ms at 16kHz
const EOU_MARKER: &str = "<EOU>";

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Event {
    Partial { text: String },
    Eou { text: String },
    Error { message: String },
}

pub struct Transcriber {
    model: ParakeetEOU,
    buffer: Vec<f32>,
    transcript: String,
    metrics: std::sync::Arc<Metrics>,
}

impl Transcriber {
    pub fn new(model_dir: impl AsRef<Path>, metrics: std::sync::Arc<Metrics>) -> Result<Self> {
        let model = ParakeetEOU::from_pretrained(model_dir, None)?;
        Ok(Self {
            model,
            buffer: Vec::with_capacity(CHUNK_SIZE * 4),
            transcript: String::new(),
            metrics,
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> Vec<Event> {
        self.buffer.extend_from_slice(samples);
        let mut events = Vec::new();
        while self.buffer.len() >= CHUNK_SIZE {
            let chunk: Vec<f32> = self.buffer.drain(..CHUNK_SIZE).collect();
            match self.run_chunk(&chunk) {
                Ok(mut evs) => events.append(&mut evs),
                Err(e) => {
                    events.push(Event::Error {
                        message: format!("inference failed: {e}"),
                    });
                }
            }
        }
        events
    }

    /// Flush any remaining buffered audio by zero-padding to a full chunk.
    /// Called on EOS from client or on disconnect.
    pub fn flush(&mut self) -> Vec<Event> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let mut chunk = std::mem::take(&mut self.buffer);
        chunk.resize(CHUNK_SIZE, 0.0);
        match self.run_chunk(&chunk) {
            Ok(mut evs) => {
                // Emit any residual transcript as a final EOU so downstream always sees a terminator.
                if !self.transcript.trim().is_empty() {
                    evs.push(Event::Eou {
                        text: std::mem::take(&mut self.transcript).trim().to_string(),
                    });
                }
                evs
            }
            Err(e) => vec![Event::Error {
                message: format!("flush inference failed: {e}"),
            }],
        }
    }

    fn run_chunk(&mut self, chunk: &[f32]) -> Result<Vec<Event>> {
        let t0 = Instant::now();
        let raw = self.model.transcribe(chunk, true)?;
        let elapsed = t0.elapsed();
        self.metrics.record_chunk(elapsed);

        let mut events = Vec::new();
        let had_eou = raw.contains(EOU_MARKER);
        let cleaned = raw.replace(EOU_MARKER, "");

        if !cleaned.is_empty() {
            append_with_spacing(&mut self.transcript, &cleaned);
        }

        if had_eou {
            let text = std::mem::take(&mut self.transcript).trim().to_string();
            if !text.is_empty() {
                events.push(Event::Eou { text });
            }
        } else if !cleaned.trim().is_empty() {
            events.push(Event::Partial {
                text: self.transcript.trim().to_string(),
            });
        }
        Ok(events)
    }
}

fn append_with_spacing(acc: &mut String, piece: &str) {
    let piece_trim = piece.trim_matches(|c: char| c == '\n' || c == '\r');
    if piece_trim.is_empty() {
        return;
    }
    if acc.is_empty() {
        acc.push_str(piece_trim.trim_start());
        return;
    }
    let ends_ws = acc.ends_with(|c: char| c.is_whitespace());
    let starts_ws = piece_trim.starts_with(|c: char| c.is_whitespace());
    if !ends_ws && !starts_ws {
        acc.push(' ');
    }
    acc.push_str(piece_trim);
}
