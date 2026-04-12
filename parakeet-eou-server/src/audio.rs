use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Deserialize;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ClientMessage {
    Audio {
        data: String,
        #[serde(default = "default_sample_rate")]
        sample_rate: u32,
        #[serde(default)]
        encoding: Option<String>,
    },
    Eos,
}

fn default_sample_rate() -> u32 {
    TARGET_SAMPLE_RATE
}

pub fn decode_binary_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(anyhow!(
            "binary audio payload length {} is not a multiple of 4 (expected little-endian f32)",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

pub fn decode_json_audio(data: &str, sample_rate: u32, encoding: Option<&str>) -> Result<Vec<f32>> {
    if sample_rate != TARGET_SAMPLE_RATE {
        return Err(anyhow!(
            "unsupported sample_rate {}: expected {} (server does not resample)",
            sample_rate,
            TARGET_SAMPLE_RATE
        ));
    }
    let raw = STANDARD
        .decode(data.as_bytes())
        .map_err(|e| anyhow!("invalid base64 audio payload: {e}"))?;

    match encoding.unwrap_or("f32le") {
        "f32le" | "f32" | "float32" => decode_binary_f32(&raw),
        "i16le" | "i16" | "pcm_s16le" => decode_i16le(&raw),
        other => Err(anyhow!("unsupported encoding '{other}' (expected f32le or i16le)")),
    }
}

fn decode_i16le(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        return Err(anyhow!(
            "i16 audio payload length {} is not a multiple of 2",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(s as f32 / 32768.0);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_f32_roundtrip() {
        let samples = [0.0f32, 0.5, -0.5, 1.0];
        let mut bytes = Vec::new();
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let decoded = decode_binary_f32(&bytes).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn rejects_bad_length() {
        assert!(decode_binary_f32(&[0u8; 3]).is_err());
    }

    #[test]
    fn rejects_non_16khz_json() {
        let b64 = STANDARD.encode([0u8; 4]);
        assert!(decode_json_audio(&b64, 48_000, None).is_err());
    }

    #[test]
    fn decodes_i16() {
        let samples: [i16; 2] = [16384, -16384];
        let mut bytes = Vec::new();
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let decoded = decode_i16le(&bytes).unwrap();
        assert!((decoded[0] - 0.5).abs() < 1e-4);
        assert!((decoded[1] + 0.5).abs() < 1e-4);
    }
}
