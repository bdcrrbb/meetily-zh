// Speaker diarization bench.
use crate::Cli;
use anyhow::Result;
use serde::Serialize;
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig, Wave,
};
use std::time::Instant;

#[derive(Serialize)]
struct Turn {
    start: f64,
    end: f64,
    speaker: i32,
}

#[derive(Serialize)]
struct Out {
    file: String,
    duration_s: f64,
    elapsed_s: f64,
    rtf: f64,
    num_speakers: i32,
    turns: Vec<Turn>,
}

pub fn run(cli: &Cli, wav_path: &str, num_clusters: i32, threshold: f32, out: &str) -> Result<()> {
    let model_dir = &cli.model_dir;
    let seg = format!(
        "{model_dir}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    );
    let emb = format!(
        "{model_dir}/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    );
    for p in [&seg, &emb] {
        if !std::path::Path::new(p).exists() {
            anyhow::bail!("missing artifact: {p}");
        }
    }
    let cfg = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(seg),
                window_shift_ratio: 0.1,
            },
            ..Default::default()
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some(emb),
            num_threads: cli.threads.max(1),
            ..Default::default()
        },
        clustering: FastClusteringConfig {
            num_clusters,
            threshold,
        },
        min_duration_on: 0.3,
        min_duration_off: 0.5,
        ..Default::default()
    };
    let t_init = Instant::now();
    let sd = OfflineSpeakerDiarization::create(&cfg).ok_or_else(|| anyhow::anyhow!("diarization init failed"))?;
    eprintln!("diarization init: {:.3}s", t_init.elapsed().as_secs_f64());

    let wave = Wave::read(wav_path).ok_or_else(|| anyhow::anyhow!("cannot read {wav_path}"))?;
    let dur = wave.samples().len() as f64 / wave.sample_rate() as f64;
    let t = Instant::now();
    let result = sd.process(wave.samples()).ok_or_else(|| anyhow::anyhow!("diarization process failed"))?;
    let elapsed = t.elapsed().as_secs_f64();

    let turns: Vec<Turn> = result
        .sort_by_start_time()
        .into_iter()
        .map(|s| Turn {
            start: s.start as f64,
            end: s.end as f64,
            speaker: s.speaker,
        })
        .collect();
    let out_v = Out {
        file: wav_path.into(),
        duration_s: dur,
        elapsed_s: elapsed,
        rtf: elapsed / dur,
        num_speakers: result.num_speakers(),
        turns,
    };
    std::fs::write(out, serde_json::to_string_pretty(&out_v)?)?;
    println!(
        "{wav_path}: dur={dur:.1}s elapsed={elapsed:.1}s rtf={:.3} speakers={}",
        out_v.rtf,
        out_v.num_speakers
    );
    Ok(())
}
