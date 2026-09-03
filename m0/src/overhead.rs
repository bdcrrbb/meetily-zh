// Per-call overhead diagnosis: decode full clip vs 2s in-memory slices.
use crate::{stt, Cli};
use anyhow::Result;
use sherpa_onnx::Wave;
use std::time::Instant;

pub fn run(cli: &Cli, wav_path: &str) -> Result<()> {
    let wave = Wave::read(wav_path)
        .ok_or_else(|| anyhow::anyhow!("cannot read {wav_path}"))?;
    let (sr, samples) = crate::to_16k(&wave);
    let dur = samples.len() as f64 / sr as f64;

    let rec = stt::recognizer(cli, "")?;

    // full clip, 3 runs
    for i in 1..=3 {
        let (el, _) = stt::decode(&rec, sr, &samples);
        println!("full run{i}: {:.3}s (dur {dur:.2}s, rtf {:.3})", el, el / dur);
    }

    // 2s slices, in memory (no temp files)
    let slice = 2 * sr as usize;
    let mut total = 0.0f64;
    let mut n = 0;
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + slice).min(samples.len());
        let (el, _) = stt::decode(&rec, sr, &samples[start..end]);
        total += el;
        n += 1;
        println!("slice {n} ({}..{end}): {el:.3}s", start / sr as usize);
        start = end;
    }
    println!(
        "slices: n={n} total={total:.3}s vs full={:.3}s | per-call overhead est: {:.3}s",
        total,
        (total - dur * 0.25) / n as f64
    );
    Ok(())
}
