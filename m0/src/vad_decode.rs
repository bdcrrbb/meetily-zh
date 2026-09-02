// Long-file decode: Silero VAD segmentation + per-segment decode with timestamps.
use crate::{stt::recognizer, vad_config, Cli};
use anyhow::Result;
use serde::Serialize;
use sherpa_onnx::{VoiceActivityDetector, Wave};

#[derive(Serialize)]
struct Seg {
    start_s: f64,
    end_s: f64,
    decode_s: f64,
    text: String,
}

#[derive(Serialize)]
struct Out {
    file: String,
    duration_s: f64,
    total_decode_s: f64,
    effective_rtf: f64,
    segments: Vec<Seg>,
}

pub fn run(cli: &Cli, wav_path: &str, out: &str) -> Result<()> {
    let wave = Wave::read(wav_path).ok_or_else(|| anyhow::anyhow!("cannot read {wav_path}"))?;
    let sr = wave.sample_rate();
    let samples = wave.samples();
    let dur = samples.len() as f64 / sr as f64;

    let mut vad = VoiceActivityDetector::create(&vad_config(&cli.model_dir), 60.0)
        .ok_or_else(|| anyhow::anyhow!("failed to create VAD"))?;
    let rec = recognizer(cli, "")?;

    let window = 512usize; // silero window
    let mut segs: Vec<Seg> = Vec::new();
    let mut total_decode = 0.0f64;
    let mut processed = 0usize;
    let t_start = std::time::Instant::now();
    while processed + window <= samples.len() {
        let chunk = &samples[processed..processed + window];
        vad.accept_waveform(chunk);
        processed += window;
        while !vad.is_empty() {
            if let Some(seg) = vad.front() {
                let start_s = seg.start() as f64 / sr as f64;
                let n = seg.n();
                let (el, text) = crate::stt::decode(&rec, sr, seg.samples());
                total_decode += el;
                segs.push(Seg {
                    start_s,
                    end_s: start_s + n as f64 / sr as f64,
                    decode_s: el,
                    text,
                });
                vad.pop();
            } else {
                break;
            }
        }
    }
    vad.flush();
    while let Some(seg) = vad.front() {
        let start_s = seg.start() as f64 / sr as f64;
        let (el, text) = crate::stt::decode(&rec, sr, seg.samples());
        total_decode += el;
        segs.push(Seg {
            start_s,
            end_s: start_s + seg.n() as f64 / sr as f64,
            decode_s: el,
            text,
        });
        vad.pop();
    }
    let wall = t_start.elapsed().as_secs_f64();
    let out_v = Out {
        file: wav_path.into(),
        duration_s: dur,
        total_decode_s: total_decode,
        effective_rtf: wall / dur,
        segments: segs,
    };
    std::fs::write(out, serde_json::to_string_pretty(&out_v)?)?;
    println!(
        "{wav_path}: dur={dur:.1}s wall={wall:.1}s effective_rtf={:.3} segs={}",
        out_v.effective_rtf,
        out_v.segments.len()
    );
    Ok(())
}
