// diarization/hierarchical.rs
//
// Two-pass windowed diarization (M2-2b).
//
// Why: single-pass greedy clustering over per-segment embeddings fragments on
// long far-field recordings (M0 spike: 64 min -> 58-104 spk) because
// same-speaker similarity hugs the merge threshold and each mis-merge is
// permanent. At ~10 min the same pipeline is healthy (7 spk, th=0.6).
//
// Pass 1: run the stock sherpa OfflineSpeakerDiarization per ~10 min window
//         (verified config) -> local speaker turns.
// Pass 2: embed each window-speaker (mean over its longest segments), then
//         greedily match window-speaker centroids across windows via
//         SpeakerEmbeddingManager (search/add, cosine threshold). Errors stay
//         local to a window; the global layer only sees few robust centroids.
//
// Refs: docs/m0-spike-report.md §5.4.1, spec v3.2 §2.

use anyhow::{anyhow, Result};
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig,
};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct HierarchicalDiarizationConfig {
    /// Window length for pass-1 clustering (spike-verified healthy scale).
    pub window_secs: f32,
    /// Pass-1 threshold. COSINE DISTANCE semantics (higher = merge more).
    /// 0.6 verified on the M0 spike corpus.
    pub window_threshold: f32,
    /// Pass-2 centroid match threshold. COSINE DISTANCE semantics, same as
    /// window_threshold (higher = merge more). Converted internally to the
    /// similarity scale where needed (similarity = 1 - distance).
    /// NOTE: sherpa's SpeakerEmbeddingManager::search uses SIMILARITY
    /// semantics - do not pass this value to it directly.
    pub merge_threshold: f32,
    /// Max segments per window-speaker used to build its centroid embedding.
    pub centroid_max_segments: usize,
    pub num_threads: i32,
}

impl Default for HierarchicalDiarizationConfig {
    fn default() -> Self {
        Self {
            window_secs: 600.0,
            window_threshold: 0.6,
            merge_threshold: 0.6,
            centroid_max_segments: 8,
            num_threads: 3,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct DiarSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

pub struct HierarchicalDiarization {
    inner: OfflineSpeakerDiarization,
    embedder: SpeakerEmbeddingExtractor,
    config: HierarchicalDiarizationConfig,
}

impl HierarchicalDiarization {
    pub fn new(models_dir: &Path, config: HierarchicalDiarizationConfig) -> Result<Self> {
        let seg = models_dir.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx");
        let emb = models_dir.join("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx");
        for p in [&seg, &emb] {
            if !p.is_file() {
                return Err(anyhow!("diarization artifact missing: {}", p.display()));
            }
        }
        let inner = OfflineSpeakerDiarization::create(&OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(seg.to_string_lossy().into()),
                    window_shift_ratio: 0.1,
                },
                ..Default::default()
            },
            embedding: SpeakerEmbeddingExtractorConfig {
                model: Some(emb.to_string_lossy().into()),
                num_threads: config.num_threads.max(1),
                ..Default::default()
            },
            clustering: FastClusteringConfig {
                num_clusters: -1,
                threshold: config.window_threshold,
            },
            min_duration_on: 0.3,
            min_duration_off: 0.5,
            ..Default::default()
        })
        .ok_or_else(|| anyhow!("failed to init sherpa diarization"))?;
        let embedder = SpeakerEmbeddingExtractor::create(&SpeakerEmbeddingExtractorConfig {
            model: Some(emb.to_string_lossy().into()),
            num_threads: config.num_threads.max(1),
            ..Default::default()
        })
        .ok_or_else(|| anyhow!("failed to init speaker embedding extractor"))?;
        Ok(Self { inner, embedder, config })
    }

    /// Diarize 16 kHz mono samples. Blocking; call via spawn_blocking.
    pub fn process(&self, samples: &[f32]) -> Result<Vec<DiarSegment>> {
        let sr = self.inner.sample_rate();
        if sr != 16000 {
            return Err(anyhow!("expected 16 kHz input, engine sample rate is {sr}"));
        }
        let window_samples = (self.config.window_secs * 16000.0) as usize;
        // Own centroid table (not sherpa's SpeakerEmbeddingManager): cosine
        // distance computed here so both passes share distance semantics and
        // rejected matches can be logged with their best score.
        let mut centroids: Vec<(usize, Vec<f32>)> = Vec::new();

        let mut out: Vec<DiarSegment> = Vec::new();
        let mut next_global = 0usize;

        for (wi, window) in samples.chunks(window_samples).enumerate() {
            let window_start_s = (wi * window_samples) as f64 / 16000.0;
            let result = self
                .inner
                .process(window)
                .ok_or_else(|| anyhow!("window {wi} diarization failed"))?;
            let segments = result.sort_by_start_time();
            if segments.is_empty() {
                continue;
            }

            // local speaker -> global id mapping for this window
            let mut mapping: Vec<Option<usize>> = Vec::new();
            let local_count = segments.iter().map(|s| s.speaker).max().unwrap_or(-1) as usize + 1;
            mapping.resize(local_count, None);

            // build centroid embeddings (mean over the longest segments per speaker)
            let mut per_speaker: Vec<Vec<&sherpa_onnx::OfflineSpeakerDiarizationSegment>> =
                vec![Vec::new(); local_count];
            for s in &segments {
                per_speaker[s.speaker as usize].push(s);
            }
            for (ls, segs) in per_speaker.iter().enumerate() {
                if segs.is_empty() {
                    continue;
                }
                let centroid = self.speaker_centroid(window, segs)?;
                let Some(centroid) = centroid else { continue };
                let best = centroids
                    .iter()
                    .map(|(id, c)| (*id, cosine_distance(&centroid, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                match best {
                    Some((id, dist)) if dist <= self.config.merge_threshold => {
                        log::debug!(
                            "hier: window {wi} local speaker {ls} -> global {id} (distance {dist:.3})"
                        );
                        mapping[ls] = Some(id);
                    }
                    Some((_, dist)) => {
                        log::debug!(
                            "hier: window {wi} local speaker {ls} rejected best match                              (distance {dist:.3} > {}), new global {next_global}",
                            self.config.merge_threshold
                        );
                        centroids.push((next_global, centroid));
                        mapping[ls] = Some(next_global);
                        next_global += 1;
                    }
                    None => {
                        centroids.push((next_global, centroid));
                        mapping[ls] = Some(next_global);
                        next_global += 1;
                    }
                }
            }

            for s in segments {
                if let Some(g) = mapping[s.speaker as usize] {
                    out.push(DiarSegment {
                        start: window_start_s + s.start as f64,
                        end: window_start_s + s.end as f64,
                        speaker: g,
                    });
                }
            }
        }

        out.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
        let distinct = out.iter().map(|s| s.speaker).collect::<std::collections::HashSet<_>>().len();
        log::info!(
            "hier: {} windows, {} clusters created, {} speakers with segments (final)",
            samples.len().div_ceil(window_samples),
            next_global,
            distinct
        );
        Ok(out)
    }

    /// Mean embedding over the speaker's longest segments (None if nothing embeddable).
    fn speaker_centroid(
        &self,
        window: &[f32],
        segs: &[&sherpa_onnx::OfflineSpeakerDiarizationSegment],
    ) -> Result<Option<Vec<f32>>> {
        let mut sorted: Vec<_> = segs.to_vec();
        sorted.sort_by(|a, b| {
            (b.end - b.start)
                .partial_cmp(&(a.end - a.start))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let chosen: Vec<_> = sorted.into_iter().take(self.config.centroid_max_segments).collect();
        let mut sum: Option<Vec<f32>> = None;
        let mut n = 0usize;
        for s in chosen {
            let start = (s.start * 16000.0) as usize;
            let end = ((s.end * 16000.0) as usize)
                .min(start + 30 * 16000) // cap 30 s per segment
                .min(window.len());
            if end <= start || end > window.len() {
                continue;
            }
            let audio = &window[start..end];
            let stream = self
                .embedder
                .create_stream()
                .ok_or_else(|| anyhow!("embedder stream"))?;
            stream.accept_waveform(16000, &audio);
            if !self.embedder.is_ready(&stream) {
                continue;
            }
            if let Some(e) = self.embedder.compute(&stream) {
                sum = Some(match sum {
                    None => e,
                    Some(mut acc) => {
                        for (a, v) in acc.iter_mut().zip(e.iter()) {
                            *a += v;
                        }
                        acc
                    }
                });
                n += 1;
            }
        }
        Ok(sum.map(|mut acc| {
            for v in acc.iter_mut() {
                *v /= n as f32;
            }
            acc
        }))
    }

}

/// Cosine distance (1 - cosine similarity) between two equal-length vectors.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - (dot / denom)
}

#[cfg(test)]
mod tests {
    // Unit-testable pieces (window math, mapping) are exercised via process()
    // structure; numeric behavior requires model artifacts (Mac/CI e2e).
    #[test]
    fn config_defaults_match_spike() {
        let c = super::HierarchicalDiarizationConfig::default();
        assert_eq!(c.window_secs, 600.0);
        assert!((c.window_threshold - 0.6).abs() < f32::EPSILON);
        assert!((c.merge_threshold - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_distance_identity_and_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        assert!(super::cosine_distance(&a, &a.clone()).abs() < 1e-6);
        let b = vec![0.0, 1.0, 0.0];
        assert!((super::cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
        // similarity 0.5 <=> distance 0.5
        let c = vec![1.0, 1.0, 0.0];
        assert!((super::cosine_distance(&a, &c) - 0.5).abs() < 1e-6);
    }
}
