// M0 bench — confirmation-gate harness for Meetily-ZH.
// Pure Rust. Subcommands: stt, vad-decode, diarize, soak, cer, report.
//
// Spec: docs/superpowers/specs/2026-09-01-meetily-zh-design.md (v3.2)
// Plan: docs/superpowers/plans/2026-09-01-meetily-zh-implementation.md (v2)

mod cer;
mod diarize;
mod soak;
mod stt;
mod vad_decode;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "m0", about = "Meetily-ZH M0 confirmation-gate harness")]
pub struct Cli {
    /// Model root dir (contains sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/ etc.)
    #[arg(long, default_value = "models")]
    pub model_dir: String,

    /// Number of ONNX Runtime threads for the recognizer
    #[arg(long, default_value_t = 3)]
    pub threads: i32,

    /// max_new_tokens for Qwen3 ASR
    #[arg(long, default_value_t = 512)]
    pub max_new_tokens: i32,

    /// max_total_len for Qwen3 ASR
    #[arg(long, default_value_t = 1024)]
    pub max_total_len: i32,

    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Decode wav(s) N runs, median RTF + text
    Stt {
        #[arg(long, num_args = 1..)]
        wavs: Vec<String>,
        #[arg(long, default_value = "m0/out/stt.json")]
        out: String,
        /// Optional hotwords, comma separated
        #[arg(long, default_value = "")]
        hotwords: String,
    },
    /// Long-file decode: Silero VAD segmentation + per-segment decode
    VadDecode {
        #[arg(long)]
        wav: String,
        #[arg(long, default_value = "m0/out/vad_decode.json")]
        out: String,
    },
    /// Speaker diarization on a wav
    Diarize {
        #[arg(long)]
        wav: String,
        /// -1 = auto via cluster threshold
        #[arg(long, default_value_t = -1)]
        num_clusters: i32,
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,
        #[arg(long, default_value = "m0/out/diarize.json")]
        out: String,
    },
    /// Dual-track real-time thermal soak
    Soak {
        #[arg(long)]
        track_a: String,
        #[arg(long)]
        track_b: String,
        #[arg(long, default_value_t = 30)]
        minutes: u64,
        #[arg(long, default_value = "m0/out/soak")]
        out_prefix: String,
    },
    /// Normalized CER: references json {clip: text} vs hyp json {clip: text}
    Cer {
        #[arg(long)]
        refs: String,
        #[arg(long)]
        hyps: String,
        #[arg(long, default_value = "m0/out/cer.json")]
        out: String,
    },
}

pub fn qwen_paths(model_dir: &str) -> anyhow::Result<sherpa_onnx::OfflineQwen3ASRModelConfig> {
    let base = format!("{model_dir}/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25");
    for f in [
        "conv_frontend.onnx",
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "tokenizer/vocab.json",
    ] {
        let p = format!("{base}/{f}");
        if !std::path::Path::new(&p).exists() {
            anyhow::bail!("missing model artifact: {p}");
        }
    }
    Ok(sherpa_onnx::OfflineQwen3ASRModelConfig {
        conv_frontend: Some(format!("{base}/conv_frontend.onnx")),
        encoder: Some(format!("{base}/encoder.int8.onnx")),
        decoder: Some(format!("{base}/decoder.int8.onnx")),
        tokenizer: Some(format!("{base}/tokenizer")),
        ..Default::default()
    })
}

pub fn vad_config(model_dir: &str) -> sherpa_onnx::VadModelConfig {
    sherpa_onnx::VadModelConfig {
        silero_vad: sherpa_onnx::SileroVadModelConfig {
            model: Some(format!("{model_dir}/silero_vad.onnx")),
            threshold: 0.2,
            min_silence_duration: 0.5,
            min_speech_duration: 0.2,
            max_speech_duration: 10.0,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    std::fs::create_dir_all("m0/out")?;
    match &cli.cmd {
        Cmd::Stt { wavs, out, hotwords } => stt::run(&cli, wavs, hotwords, out)?,
        Cmd::VadDecode { wav, out } => vad_decode::run(&cli, wav, out)?,
        Cmd::Diarize { wav, num_clusters, threshold, out } => {
            diarize::run(&cli, wav, *num_clusters, *threshold, out)?
        }
        Cmd::Soak { track_a, track_b, minutes, out_prefix } => {
            soak::run(&cli, track_a, track_b, *minutes, out_prefix)?
        }
        Cmd::Cer { refs, hyps, out } => cer::run(refs, hyps, out)?,
    }
    Ok(())
}
