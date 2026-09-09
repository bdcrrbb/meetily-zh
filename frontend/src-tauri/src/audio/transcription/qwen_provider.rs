// audio/transcription/qwen_provider.rs
//
// Qwen3-ASR (sherpa-onnx int8 ONNX) transcription provider.
//
// Design refs:
//   docs/superpowers/specs/2026-09-01-meetily-zh-design.md (v3.2, Plan A)
//   docs/notes-engine-map.md
//
// Invariants:
// - NEVER decode unbounded audio in one call (OOM observed on 64-min wav).
//   Inputs longer than MAX_SINGLE_SHOT_SECS are VAD-segmented and decoded
//   in pieces of <= MAX_PIECE_SECS, then stitched.
// - The recognizer is owned by this provider; the transcription worker is
//   single-flight (NUM_WORKERS=1), so all calls are serialized by design.
// - Per-call fixed overhead measured at ~0.37s on M2 Air (m0 overhead cmd).

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use log::{info, warn};
use sherpa_onnx::{
    OfflineModelConfig, OfflineQwen3ASRModelConfig, OfflineRecognizer,
    OfflineRecognizerConfig, VoiceActivityDetector,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const MAX_SINGLE_SHOT_SECS: f64 = 300.0; // import path guard: VAD-segment beyond this
const MAX_PIECE_SECS: f64 = 10.0; // hard piece cap (matches m0 force-split findings)
const VAD_BUFFER_SECS: f32 = 120.0;
const DEFAULT_MAX_NEW_TOKENS: i32 = 512;
const DEFAULT_MAX_TOTAL_LEN: i32 = 1024;

pub const QWEN3_MODEL_DIR_NAME: &str = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25";
pub const QWEN3_MODEL_NAME: &str = "qwen3-asr-0.6B-int8";

/// Required artifact paths, relative to the models dir.
pub fn required_artifacts() -> [PathBuf; 4] {
    let base = Path::new(QWEN3_MODEL_DIR_NAME);
    [
        base.join("conv_frontend.onnx"),
        base.join("encoder.int8.onnx"),
        base.join("decoder.int8.onnx"),
        base.join("tokenizer/vocab.json"),
    ]
}

/// Preflight: all artifacts present under `models_dir`?
pub fn artifacts_present(models_dir: &Path) -> bool {
    required_artifacts()
        .iter()
        .all(|p| models_dir.join(p).is_file())
}

fn qwen_model_config(models_dir: &Path) -> OfflineQwen3ASRModelConfig {
    let base = models_dir.join(QWEN3_MODEL_DIR_NAME);
    OfflineQwen3ASRModelConfig {
        conv_frontend: Some(base.join("conv_frontend.onnx").to_string_lossy().into()),
        encoder: Some(base.join("encoder.int8.onnx").to_string_lossy().into()),
        decoder: Some(base.join("decoder.int8.onnx").to_string_lossy().into()),
        tokenizer: Some(base.join("tokenizer").to_string_lossy().into()),
        max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        max_total_len: DEFAULT_MAX_TOTAL_LEN,
        ..Default::default()
    }
}

fn vad_config(models_dir: &Path) -> sherpa_onnx::VadModelConfig {
    sherpa_onnx::VadModelConfig {
        silero_vad: sherpa_onnx::SileroVadModelConfig {
            model: Some(models_dir.join("silero_vad.onnx").to_string_lossy().into()),
            threshold: 0.2,
            min_silence_duration: 0.5,
            min_speech_duration: 0.2,
            max_speech_duration: MAX_PIECE_SECS as f32,
            window_size: 512,
            ..Default::default()
        },
        sample_rate: 16000,
        num_threads: 1,
        provider: Some("cpu".into()),
        debug: false,
        ..Default::default()
    }
}

pub struct Qwen3Provider {
    recognizer: OfflineRecognizer,
    models_dir: PathBuf,
}

impl Qwen3Provider {
    /// Create the provider, verifying artifacts first.
    pub fn new(models_dir: &Path, num_threads: i32) -> Result<Self, String> {
        if !artifacts_present(models_dir) {
            let missing: Vec<String> = required_artifacts()
                .iter()
                .filter(|p| !models_dir.join(p).is_file())
                .map(|p| p.to_string_lossy().into())
                .collect();
            return Err(format!(
                "Qwen3-ASR model artifacts missing under {}: {}",
                models_dir.display(),
                missing.join(", ")
            ));
        }
        let mut cfg = OfflineRecognizerConfig::default();
        cfg.model_config = OfflineModelConfig {
            qwen3_asr: qwen_model_config(models_dir),
            tokens: Some(String::new()),
            provider: Some("cpu".into()),
            num_threads,
            debug: false,
            ..Default::default()
        };
        let recognizer = OfflineRecognizer::create(&cfg)
            .ok_or_else(|| "Failed to create Qwen3-ASR recognizer (model load failed)".to_string())?;
        info!("✅ Qwen3-ASR recognizer created (models dir: {})", models_dir.display());
        Ok(Self {
            recognizer,
            models_dir: models_dir.to_path_buf(),
        })
    }

    fn decode_samples(&self, sr: i32, samples: &[f32]) -> String {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(sr, samples);
        self.recognizer.decode(&stream);
        stream
            .get_result()
            .map(|r| r.text)
            .unwrap_or_default()
    }

    /// VAD-segment `samples` (16 kHz mono) and decode piecewise.
    fn vad_segmented_decode(&self, samples: &[f32]) -> Result<String, TranscriptionError> {
        let vad = VoiceActivityDetector::create(&vad_config(&self.models_dir), VAD_BUFFER_SECS)
            .ok_or_else(|| {
                TranscriptionError::EngineFailed("Failed to create VAD for long input".into())
            })?;
        const WINDOW: usize = 512;
        let mut pieces: Vec<String> = Vec::new();
        let mut processed = 0usize;
        while processed + WINDOW <= samples.len() {
            vad.accept_waveform(&samples[processed..processed + WINDOW]);
            processed += WINDOW;
            while !vad.is_empty() {
                if let Some(seg) = vad.front() {
                    // hard force-split: MAX_PIECE_SECS pieces (VAD does not
                    // enforce max_speech_duration on continuous speech)
                    for piece in seg.samples().chunks(MAX_PIECE_SECS as usize * 16000) {
                        pieces.push(self.decode_samples(16000, piece));
                    }
                    vad.pop();
                } else {
                    break;
                }
            }
        }
        vad.flush();
        while let Some(seg) = vad.front() {
            for piece in seg.samples().chunks(MAX_PIECE_SECS as usize * 16000) {
                pieces.push(self.decode_samples(16000, piece));
            }
            vad.pop();
        }
        Ok(stitch(&pieces))
    }
}

/// Stitch piece texts. No separator when either side is CJK (Chinese
/// typography: no space between CJK and digits/latin); a single space is
/// inserted only between two latin/alphanumeric boundaries. Display-layer
/// spacing (盘古之白) is a rendering concern, handled outside transcription.
fn stitch(pieces: &[String]) -> String {
    let mut out = String::new();
    for p in pieces {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        if let (Some(last), Some(first)) = (out.chars().last(), p.chars().next()) {
            if last.is_ascii_alphanumeric() && first.is_ascii_alphanumeric() {
                out.push(' ');
            }
        }
        out.push_str(p);
    }
    out
}

#[async_trait]
impl TranscriptionProvider for Qwen3Provider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        if let Some(ref lang) = language {
            // Qwen3 auto-detects language; the hint is informational only.
            warn!("Qwen3-ASR auto-detects language; ignoring hint '{}'", lang);
        }
        let dur = audio.len() as f64 / 16000.0;
        let text = if dur > MAX_SINGLE_SHOT_SECS {
            info!(
                "Qwen3: input {:.0}s > {:.0}s, using VAD-segmented decode",
                dur, MAX_SINGLE_SHOT_SECS
            );
            self.vad_segmented_decode(&audio)?
        } else {
            self.decode_samples(16000, &audio)
        };
        Ok(TranscriptResult {
            text: text.trim().to_string(),
            confidence: None,
            is_partial: false,
        })
    }

    async fn is_model_loaded(&self) -> bool {
        true // recognizer is created eagerly in new(); lifetime = provider
    }

    async fn get_current_model(&self) -> Option<String> {
        Some(QWEN3_MODEL_NAME.to_string())
    }

    fn provider_name(&self) -> &'static str {
        "Qwen3"
    }
}

// ---------------------------------------------------------------------------
// Shared engine instance (created once, reused across recording tasks)
// ---------------------------------------------------------------------------

static QWEN3_ENGINE: Mutex<Option<Arc<Qwen3Provider>>> = Mutex::new(None);

/// Get or create the shared Qwen3 provider. `models_dir` defaults to the
/// app's models directory when None.
pub fn get_or_init_qwen3_provider(
    models_dir: Option<&Path>,
    num_threads: i32,
) -> Result<Arc<Qwen3Provider>, String> {
    let mut guard = QWEN3_ENGINE
        .lock()
        .map_err(|_| "Qwen3 provider lock poisoned".to_string())?;
    if let Some(existing) = guard.as_ref() {
        return Ok(existing.clone());
    }
    let dir = match models_dir {
        Some(d) => d.to_path_buf(),
        None => crate::paths::models_dir(),
    };
    let provider = Arc::new(Qwen3Provider::new(&dir, num_threads)?);
    *guard = Some(provider.clone());
    Ok(provider)
}

#[cfg(test)]
mod tests {
    use super::stitch;

    #[test]
    fn stitch_cjk_no_space() {
        assert_eq!(stitch(&["你好".into(), "世界".into()]), "你好世界");
    }

    #[test]
    fn stitch_latin_adds_space() {
        assert_eq!(stitch(&["hello".into(), "world".into()]), "hello world");
    }

    #[test]
    fn stitch_mixed_and_empty() {
        // CJK<->digit boundaries take no separator (Chinese typography;
        // matches Qwen3 native output style). Display-layer 盘古之白 is
        // handled by the normalization layer, not transcription.
        assert_eq!(stitch(&["结果是".into(), "".into(), "150".into()]), "结果是150");
        assert_eq!(stitch(&["report".into(), "2024".into()]), "report 2024");
        assert_eq!(stitch(&[]), "");
    }
}
