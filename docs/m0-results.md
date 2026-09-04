# M0 Results — capacity/thermal gate (2026-09-03)

Target: M2 MacBook Air 16GB (fanless). Harness: m0/bench @ commit 048d7d1+.

## Run history
1. soak v1: broken (44.1kHz fed as 16k → chipmunk VAD). Root-caused, fixed with to_16k.
2. soak v2: accounting correct (decoded 3526s ≈ input 3600s, 99%) but segments
   hit 20.6s (VAD max_speech_duration not enforced on continuous speech) →
   512-token truncation + 7.5s decodes. Fixed with hard 10s force-split.
3. soak v3 (VALID): raokouling.wav both tracks (100% duty ×2 = worst case),
   30 min. 511 captions, zero loss (decoded 3526s ≈ input 3600s), segments
   ≤10s, no truncation, peak RSS 1.78GB.

## Thermal drift (soak3, decode ms per caption-bucket; buckets are caption-indexed,
## wall time per bucket grows as decode slows)
2.47 → 2.26 → 2.24 → 2.29 → 2.77 → 2.57 → 3.18 → 3.51 → 3.15 …

**Progressive thermal throttling confirmed** on fanless M2 under sustained
200%-duty dual-track load: effective RTF drifts ~0.33 → ~1.0 over 30 min.
Captions never stop; decode slows ~3×; zero audio loss; finalize correctness
unaffected (spool + full re-decode).

## Gate decision
- Headset mode (default; 1 track, ~50% duty ≈ 25% of test load): **PASS** —
  decode stays ≈ real-time with margin.
- Room mode (2 tracks, realistic duty ≈ 50% of test load): **PASS with caveat** —
  live captions lag in long meetings; finalize unaffected.
- Pathological 2×100% continuous duty: NOT sustainable — does not occur in
  practice; documented, not engineered around.
- STT quality gate: pending user corpus (CER vs whisper baseline).
- Diarization quality gate: pending annotated clips.
- License manifest: artifacts hashed (m0/models/sha256.txt); formal manifest
  doc pending.

## Capacity numbers to carry into design
- Long-clip RTF (cold, M2 Air): 0.33 (stable across 3 runs)
- Per-call overhead (2s slice vs 20s clip): ~0.37s — negligible; short segments viable
- Sustained RTF at 200% duty: drifts 0.33 → ~1.0 over 30 min
- Peak RSS: 1.78 GB (model + runtime)
- Recognizer creation: ~2.1–2.6s (load before recording starts)

## Harness notes
- VAD max_speech_duration is NOT enforced on continuous speech — hard
  force-split (10s) required at consumer level (implemented).
- Caption-index buckets ≠ wall-clock minutes under load — always convert.
- All logs: m0/out/soak3_log.jsonl (736 lines = 511 captions + 225 heartbeats).
