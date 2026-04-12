use std::sync::Mutex;
use std::time::Duration;

const WINDOW: usize = 128;

pub struct Metrics {
    chunk_latencies_ms: Mutex<Vec<f64>>,
    pub model_load_time_ms: f64,
}

impl Metrics {
    pub fn new(model_load_time_ms: f64) -> Self {
        Self {
            chunk_latencies_ms: Mutex::new(Vec::with_capacity(WINDOW)),
            model_load_time_ms,
        }
    }

    pub fn record_chunk(&self, d: Duration) {
        let ms = d.as_secs_f64() * 1000.0;
        let mut v = self.chunk_latencies_ms.lock().unwrap();
        if v.len() >= WINDOW {
            v.remove(0);
        }
        v.push(ms);
    }

    pub fn avg_latency_ms(&self) -> f64 {
        let v = self.chunk_latencies_ms.lock().unwrap();
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }
}
