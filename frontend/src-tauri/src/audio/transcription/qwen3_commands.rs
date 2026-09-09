// audio/transcription/qwen3_commands.rs
//
// Tauri commands for Qwen3-ASR model status + download (artifact manifest:
// docs/model-manifest.md). Download = pinned GitHub release URLs, streamed
// to the models dir with progress events, then bzip2/tar extraction + sha256
// verification (M0 sha256.txt values).

use anyhow::Result;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;
use log::info;
use std::time::Instant;
use tauri::{AppHandle, Emitter};

use super::qwen_provider::{artifacts_present, QWEN3_MODEL_DIR_NAME};

const SILERO_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
const QWEN3_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2";
const PYANNOTE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
const ERES2NET_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";

const SILERO_SHA: &str = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6";
const QWEN3_CONV_SHA: &str = "d22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e";
const QWEN3_DEC_SHA: &str = "4f6885be5959ae26af3089d38ee7972c5fafbeeb1cf8d5e76eab6d8b61ca5771";
const QWEN3_ENC_SHA: &str = "60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9";
const PYANNOTE_SHA: &str = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079";
const ERES2NET_SHA: &str = "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b";

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Qwen3Status {
    pub artifacts_present: bool,
    pub models_dir: String,
    pub missing: Vec<String>,
}

fn missing_list(models_dir: &Path) -> Vec<String> {
    let base = models_dir.join(QWEN3_MODEL_DIR_NAME);
    [
        base.join("conv_frontend.onnx"),
        base.join("encoder.int8.onnx"),
        base.join("decoder.int8.onnx"),
        base.join("tokenizer/vocab.json"),
        models_dir.join("silero_vad.onnx"),
    ]
    .iter()
    .filter(|p| !p.is_file())
    .map(|p| p.to_string_lossy().into())
    .collect()
}

#[tauri::command]
pub fn qwen3_status() -> Qwen3Status {
    let dir = crate::paths::models_dir();
    let missing = missing_list(&dir);
    Qwen3Status {
        artifacts_present: missing.is_empty() && artifacts_present(&dir),
        models_dir: dir.to_string_lossy().into(),
        missing,
    }
}

fn emit_progress(app: &AppHandle, stage: &str, received: u64, total: u64) {
    let _ = app.emit(
        "qwen3-download-progress",
        serde_json::json!({ "stage": stage, "received": received, "total": total }),
    );
}

async fn download_file(app: &AppHandle, url: &str, dest: &Path, stage: &str) -> Result<()> {
    if dest.exists() {
        info!("qwen3 download: {} exists, skipping", dest.display());
        return Ok(());
    }
    emit_progress(app, stage, 0, 0);
    let resp = reqwest::get(url).await?.error_for_status()?;
    let total = resp.content_length().unwrap_or(0);
    let mut file = std::fs::File::create(dest)?;
    let mut stream = resp.bytes_stream();
    let mut received = 0u64;
    let mut last_emit = Instant::now();
    use futures_util::StreamExt;
    use std::io::Write;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        received += chunk.len() as u64;
        if last_emit.elapsed().as_millis() > 300 {
            emit_progress(app, stage, received, total);
            last_emit = Instant::now();
        }
    }
    file.flush()?;
    emit_progress(app, stage, received, total);
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(format!("{:x}", h.finalize()))
}

fn verify(path: &Path, expected: &str) -> Result<()> {
    let got = sha256_file(path)?;
    if got != expected {
        std::fs::remove_file(path).ok();
        anyhow::bail!("checksum mismatch for {}: got {}", path.display(), got);
    }
    Ok(())
}

fn extract_tar_bz2(archive: &Path, dest: &Path) -> Result<()> {
    let f = std::fs::File::open(archive)?;
    let d = bzip2::read::BzDecoder::new(f);
    tar::Archive::new(d).unpack(dest)?;
    Ok(())
}

/// Download all Qwen3-path artifacts (ASR + VAD + diarization prerequisites).
/// Sequential; resumable per-file (existing complete files are skipped).
#[tauri::command]
pub async fn qwen3_download(app: AppHandle) -> Result<Qwen3Status, String> {
    let models_dir = crate::paths::models_dir();
    std::fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;
    let tmp = models_dir.join("_qwen3_tmp");
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;

    // 1. silero VAD
    let silero = models_dir.join("silero_vad.onnx");
    download_file(&app, SILERO_URL, &silero, "silero_vad")
        .await
        .map_err(|e| e.to_string())?;
    verify(&silero, SILERO_SHA).map_err(|e| e.to_string())?;

    // 2. qwen3 ASR tarball
    let qwen_tar = tmp.join("qwen3.tar.bz2");
    download_file(&app, QWEN3_URL, &qwen_tar, "qwen3_asr")
        .await
        .map_err(|e| e.to_string())?;
    extract_tar_bz2(&qwen_tar, &models_dir).map_err(|e| e.to_string())?;
    std::fs::remove_file(&qwen_tar).ok();
    // verify extracted onnx files
    let base = models_dir.join(QWEN3_MODEL_DIR_NAME);
    verify(&base.join("conv_frontend.onnx"), QWEN3_CONV_SHA).map_err(|e| e.to_string())?;
    verify(&base.join("encoder.int8.onnx"), QWEN3_ENC_SHA).map_err(|e| e.to_string())?;
    verify(&base.join("decoder.int8.onnx"), QWEN3_DEC_SHA).map_err(|e| e.to_string())?;

    // 3. pyannote segmentation (diarization prerequisite, M2)
    let pyannote_tar = tmp.join("pyannote.tar.bz2");
    download_file(&app, PYANNOTE_URL, &pyannote_tar, "pyannote")
        .await
        .map_err(|e| e.to_string())?;
    extract_tar_bz2(&pyannote_tar, &models_dir).map_err(|e| e.to_string())?;
    std::fs::remove_file(&pyannote_tar).ok();
    verify(
        &models_dir.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
        PYANNOTE_SHA,
    )
    .map_err(|e| e.to_string())?;

    // 4. eres2net zh-cn embedding (diarization prerequisite, M2)
    let emb = models_dir.join("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx");
    download_file(&app, ERES2NET_URL, &emb, "eres2net")
        .await
        .map_err(|e| e.to_string())?;
    verify(&emb, ERES2NET_SHA).map_err(|e| e.to_string())?;

    std::fs::remove_dir_all(&tmp).ok();
    let _ = app.emit("qwen3-download-progress", serde_json::json!({ "stage": "done" }));

    let missing = missing_list(&models_dir);
    Ok(Qwen3Status {
        artifacts_present: missing.is_empty() && artifacts_present(&models_dir),
        models_dir: models_dir.to_string_lossy().into(),
        missing,
    })
}

