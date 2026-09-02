// Dual-track real-time thermal soak.
// Two producer threads replay wavs at 1x real time into bounded channels;
// a single consumer runs per-track VAD + serialized decode (production topology).
use crate::{vad_config, Cli};
use anyhow::Result;
use serde::Serialize;
use sherpa_onnx::{VoiceActivityDetector, Wave};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const CHUNK: usize = 1600; // 100ms @ 16k
const CHANNEL_CAPACITY_CHUNKS: usize = 300; // 30s per track

#[derive(Serialize, serde::Deserialize, Clone)]
struct Caption {
    track: u8,
    seq: u64,
    capture_to_text_ms: f64,
    decode_ms: f64,
    chars: usize,
}

#[derive(Serialize)]
struct SoakSummary {
    minutes: u64,
    captions: usize,
    lat_p50_ms: f64,
    lat_p95_ms: f64,
    lat_p99_ms: f64,
    peak_rss_kb: u64,
    dropped: u64,
}

fn peak_rss_kb() -> u64 {
    // Linux: /proc/self/status; macOS: getrusage via status fallback
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("VmHWM:") {
                return v.split_whitespace().next().unwrap_or("0").parse().unwrap_or(0);
            }
        }
    }
    0 // macOS: report via `ps -o rss` externally if needed
}

fn producer(path: &str, track: u8, tx: mpsc::SyncSender<(u8, u64, Instant, Vec<f32>)>, stop: std::sync::Arc<AtomicU64>) -> Result<()> {
    let wave = Wave::read(path).ok_or_else(|| anyhow::anyhow!("cannot read {path}"))?;
    let sr = wave.sample_rate();
    let samples = wave.samples();
    let mut seq = 0u64;
    let start = Instant::now();
    let mut pos = 0usize;
    while stop.load(Ordering::Relaxed) == 0 {
        // real-time pacing anchored to start
        let due = start + Duration::from_secs_f64(seq as f64 * CHUNK as f64 / sr as f64);
        let now = Instant::now();
        if due > now {
            std::thread::sleep(due - now);
        }
        let end = (pos + CHUNK).min(samples.len());
        let chunk: Vec<f32> = samples[pos..end].to_vec();
        pos = if end == samples.len() { 0 } else { end };
        if tx.send((track, seq, Instant::now(), chunk)).is_err() {
            break;
        }
        seq += 1;
    }
    Ok(())
}

pub fn run(cli: &Cli, track_a: &str, track_b: &str, minutes: u64, out_prefix: &str) -> Result<()> {
    let stop = std::sync::Arc::new(AtomicU64::new(0));
    let (tx, rx) = mpsc::sync_channel::<(u8, u64, Instant, Vec<f32>)>(CHANNEL_CAPACITY_CHUNKS);

    // Build inference objects on this thread, then move them into the consumer.
    let rec = std::sync::Mutex::new(crate::stt::recognizer(cli, "")?);
    let mut vads = [
        VoiceActivityDetector::create(&vad_config(&cli.model_dir), 120.0)
            .ok_or_else(|| anyhow::anyhow!("vad0"))?,
        VoiceActivityDetector::create(&vad_config(&cli.model_dir), 120.0)
            .ok_or_else(|| anyhow::anyhow!("vad1"))?,
    ];

    let stop_p = stop.clone();
    let pa = track_a.to_string();
    let tx_a = tx.clone();
    let th_a = std::thread::spawn(move || producer(&pa, 0, tx_a, stop_p));
    let stop_p = stop.clone();
    let pb = track_b.to_string();
    let th_b = std::thread::spawn(move || producer(&pb, 1, tx, stop_p));

    let log_path = format!("{out_prefix}_log.jsonl");
    let mut log = std::io::BufWriter::new(std::fs::File::create(&log_path)?);
    let consumer = std::thread::spawn(move || -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(minutes * 60);
        while Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(500)) {
                Ok((track, seq, capture, chunk)) => {
                    let vad = &vads[track as usize];
                    vad.accept_waveform(&chunk);
                    while !vad.is_empty() {
                        if let Some(seg) = vad.front() {
                            let (decode_ms, text) = {
                                let rec = rec.lock().unwrap();
                                let t = Instant::now();
                                let (el, text) = crate::stt::decode(&rec, 16000, seg.samples());
                                (t.elapsed().as_secs_f64() * 1000.0, text)
                            };
                            let _ = &decode_ms;
                            let cap = Caption {
                                track,
                                seq,
                                capture_to_text_ms: capture.elapsed().as_secs_f64() * 1000.0,
                                decode_ms,
                                chars: text.chars().count(),
                            };
                            serde_json::to_writer(&mut log, &cap)?;
                            writeln!(log)?;
                            vad.pop();
                        } else {
                            break;
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(_) => break,
            }
        }
        stop.store(1, Ordering::Relaxed);
        eprintln!("soak window over");
        Ok(())
    });

    consumer.join().unwrap()?;
    let _ = th_a.join();
    let _ = th_b.join();

    // collect captions for percentiles
    let mut lats: Vec<f64> = Vec::new();
    // captions were sent to txt_rx in consumer; re-read log instead (simpler)
    for line in std::fs::read_to_string(&log_path)?.lines() {
        if let Ok(c) = serde_json::from_str::<Caption>(line) {
            lats.push(c.capture_to_text_ms);
        }
    }
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 {
        if lats.is_empty() { 0.0 } else { lats[((p / 100.0) * (lats.len() - 1) as f64) as usize] }
    };
    let summary = SoakSummary {
        minutes,
        captions: lats.len(),
        lat_p50_ms: pct(50.0),
        lat_p95_ms: pct(95.0),
        lat_p99_ms: pct(99.0),
        peak_rss_kb: peak_rss_kb(),
        dropped: 0,
    };
    std::fs::write(
        format!("{out_prefix}_summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!(
        "soak: {} min, {} captions, p50={:.0}ms p95={:.0}ms p99={:.0}ms peakRSS={}kB",
        minutes, summary.captions, summary.lat_p50_ms, summary.lat_p95_ms, summary.lat_p99_ms, summary.peak_rss_kb
    );
    Ok(())
}

