# Engine Map — fork transcription internals (M1 spike)

Read 2026-09-03, target: `frontend/src-tauri/src/audio/transcription/` (+ engines).

## Call graph (live path)
```
capture (audio/) → AudioChunk → tokio unbounded channel
  → start_transcription_task (worker.rs)
      → get_or_init_transcription_engine(app)   [engine.rs, reads TranscriptConfig from db]
          → TranscriptionEngine::{Whisper|Parakeet|Provider(Arc<dyn TranscriptionProvider>)}
      → NUM_WORKERS = 1 (serial mode "guaranteeing zero chunk loss", ordered emission)
      → worker loop: recv chunk → engine transcribe → emit TranscriptUpdate
      → counters: chunks_queued / chunks_completed (AtomicU64), input_finished flag
  → TranscriptUpdate → app events → UI / recording manager (final results persisted)
```

## Key facts affecting M1
1. **Serial worker already exists** (NUM_WORKERS=1) — matches our single-flight design; no concurrency rework needed for M1/M2.
2. **Channel is `unbounded`** — the backpressure gap the plan calls out (M3: replace with bounded audio-seconds queue + spool).
3. **Vocabulary support already wired**: `VocabularyRepository::get_effective` → `initial_prompt` passed to engine — Qwen `hotwords` field plugs in here.
4. `TranscriptUpdate` carries `chunk_start_time` (legacy f64) — timestamps exist per chunk; merge can build on this.
5. Engine init reads `TranscriptConfig {provider, model, api_key}` from app DB — adding Qwen3 = new provider string + model entries in the config/model-manager UI, plus a new `TranscriptionEngine::Qwen3(Arc<Qwen3Provider>)` arm in the 3 match sites (engine.rs ×4, worker.rs ×2, transcription/mod.rs dispatch at lines ~516-593).
6. Model readiness check exists pre-recording (`validate_transcription_model_ready`) — reuse for our warm-start contract.

## Import path (M1 target)
- Import/retranscribe goes through the same engine (`RetranscribeDialog` → engine.transcribe on file samples).
- Our Qwen provider integrates via `TranscriptionProvider` trait (already designed in provider.rs, one-shot `transcribe(audio, language)`).
- M1 plan adjustment (smaller than planned): implement `Qwen3Provider` on the EXISTING trait (no new request struct yet); introduce versioned `TranscriptionRequest` only in M3 when live dual-track needs seq/track semantics. Rationale: import path doesn't need them; avoids touching all call sites twice.

## Duration guard (added to M1 task list)
Single-shot decode of long files OOM-kills (observed: 64-min wav killed the process). Import path must: VAD-segment via sherpa first, decode per ≤10s piece, stitch. Never call decode() on unbounded audio.

## Other modules noted
- `audio/incremental_saver.rs`, `encode.rs`, `ffmpeg_mixer.rs` — recording persistence + ffmpeg resolve (auto-download) — reuse for import normalization.
- `diarization/` — old Rust diarizer, untouched until M3 swap decision.
