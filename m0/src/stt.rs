// STT bench: N runs per clip, median RTF, text output.
use crate::{qwen_paths, Cli};
use anyhow::Result;
use serde::Serialize;
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, Wave,
};
use std::time::Instant;

#[derive(Serialize, Clone)]
struct SttResult {
    clip: String,
    duration_s: f64,
    creation_s: f64,
    runs_elapsed_s: Vec<f64>,
    median_rtf: f64,
    text: String,
}

pub fn recognizer(cli: &Cli, hotwords: &str) -> Result<OfflineRecognizer> {
    let mut cfg = OfflineRecognizerConfig::default();
    cfg.model_config = OfflineModelConfig {
        qwen3_asr: qwen_paths(&cli.model_dir)?,
        tokens: Some(String::new()),
        provider: Some("cpu".into()),
        num_threads: cli.threads,
        debug: false,
        ..Default::default()
    };
    cfg.model_config.qwen3_asr.max_new_tokens = cli.max_new_tokens;
    cfg.model_config.qwen3_asr.max_total_len = cli.max_total_len;
    cfg.model_config.qwen3_asr.hotwords = Some(hotwords.to_string());
    let t = Instant::now();
    let rec = OfflineRecognizer::create(&cfg).ok_or_else(|| anyhow::anyhow!("failed to create recognizer"))?;
    eprintln!("recognizer created in {:.3}s", t.elapsed().as_secs_f64());
    Ok(rec)
}

pub fn decode(rec: &OfflineRecognizer, sr: i32, samples: &[f32]) -> (f64, String) {
    let stream = rec.create_stream();
    let t = Instant::now();
    stream.accept_waveform(sr, samples);
    rec.decode(&stream);
    let elapsed = t.elapsed().as_secs_f64();
    let text = stream
        .get_result()
        .map(|r| r.text)
        .unwrap_or_default();
    (elapsed, text)
}

pub fn run(cli: &Cli, wavs: &[String], hotwords: &str, out: &str) -> Result<()> {
    let rec = recognizer(cli, hotwords)?;
    let mut results = Vec::new();
    for w in wavs {
        let wave = Wave::read(w).ok_or_else(|| anyhow::anyhow!("cannot read wav {w}"))?;
        let dur = wave.samples().len() as f64 / wave.sample_rate() as f64;
        let mut runs = Vec::new();
        let mut text = String::new();
        for _ in 0..3 {
            let (el, tx) = decode(&rec, wave.sample_rate(), wave.samples());
            runs.push(el);
            text = tx;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = runs[runs.len() / 2];
        let rtf = median / dur;
        println!(
            "{w}: dur={dur:.2}s median={median:.3}s RTF={rtf:.3} chars={}",
            text.chars().count()
        );
        results.push(SttResult {
            clip: w.clone(),
            duration_s: dur,
            creation_s: 0.0,
            runs_elapsed_s: runs,
            median_rtf: rtf,
            text,
        });
    }
    std::fs::write(out, serde_json::to_string_pretty(&results)?)?;
    eprintln!("wrote {out}");
    Ok(())
}
