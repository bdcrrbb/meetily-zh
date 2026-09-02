# Meetily-ZH Design v3 (FINAL) — Chinese-First, Fully Local Meeting Assistant

**Date:** 2026-09-01 (v3 final; v2 → v3: Plan A confirmed as the architecture, sidecar demoted to contingency)
**Status:** Design finalized; M0 confirmation gate remains before implementation
**Base:** Fork of `TylerBuza/Meetily-ActuallyFree` (MIT; itself a fork of `Zackriya-Solutions/meetily`)

---

## 0. Product Decisions (grill session 2026-09-01 — binding)

| # | Decision | Consequence |
|---|---|---|
| D1 | **Personal use only** | No signing/notarization/SBOM; M4 = Gatekeeper bypass doc + personal build script; license manifest informational (not a redistribution gate) |
| D2 | **Target: M2 MacBook Air, 16 GB** (fanless!) | Min-spec = this machine; M0 thermal soak is the highest-priority gate number; summary default = **LAN Ollama (192.168.5.198)** to keep unified memory free; Mac-local Ollama offered, fine on 16 GB |
| D3 | **Mixed meeting setups; priority = room setups + recordings** | Room-mic mode **in v1 scope** (not deferred); mic-voiceprint "You" pinning matters; headset mode remains the cheap default path |
| D4 | **Summary model: anything allowed** (LAN/cloud/local all OK) | Provider UI unchanged from upstream; default endpoint per D2 |
| D5 | **Recordings-first usability** | Milestone reorder: **M1 import provider + M2 diarization/finalize = first usable release** ("drop in recordings → attributed zh transcript + summary"); live capture becomes M3; polish M4 |
| D6 | **English UI fine; keep ActuallyFree features until they conflict** | No feature stripping in v1; conflicts resolved case-by-case |
| D7 | **Private repo; keep MIT attribution/notices** | Accept divergence from ActuallyFree upstream; cherry-pick fixes opportunistically |
| D8 | **Whisper engine cut from v1** | Removed from engine picker; not part of `TranscriptionRequest` v1 test surface (can re-add later); recovery protocol references "other engine" generically |

## 1. Goal

A macOS meeting assistant where **everything runs locally on the Mac** (audio capture, STT, diarization, storage), optimized for **Chinese meeting transcription**, with the same summary-model flexibility as upstream Meetily (local Ollama or BYOK cloud — the only optional off-device component, user-selected).

Non-goals: Windows/Linux builds (later), Enterprise features (SSO, audit trails), cloud anything.

## 2. Final Architecture — "All sherpa-onnx" (Plan A)

**One inference toolchain for everything. No Python. No MPS. No sidecar.**

| Concern | Choice |
|---|---|
| STT | **Qwen3-ASR-0.6B int8 ONNX** (`sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`), ONNX Runtime CPU, sherpa-onnx **Rust API** |
| VAD | Silero (sherpa-onnx) |
| Live STT | VAD-chunked **live phrase transcription** (segments closed on ≥500ms silence; 6–10s max deadline; 0.8s min; ±200ms padding) |
| Finalize STT | Full-track VAD-segmented decode (sherpa long-file pattern) on spooled audio |
| Diarization | **sherpa-onnx offline speaker diarization pipeline as a unit**: pyannote-segmentation-3.0 ONNX + 3D-Speaker zh-cn embedding (192-d class; exact artifact pinned at M0) + FastClustering |
| Voice profiles | New-model profiles tagged `{model_id, revision, dim}`; legacy 128-d WeSpeaker profiles never cross-compared |
| Fallback STT | whisper.cpp (existing fork provider) |
| Summary | Untouched upstream engine: Ollama / BYOK / custom OpenAI-compatible endpoint |
| Shell | Tauri (Rust) + Next.js, SQLite; macOS Apple Silicon only (v1) |

Evidence base: VoicePing benchmark (2026-02) measured Qwen3-ASR-0.6B int8 at 8.0 tok/s on an M4 (~1.6GB RSS), Android RTF 0.53 → estimated M4 RTF ~0.2–0.3; sherpa-onnx diarization on Chinese 4-speaker audio measures RTF 0.24–0.30 on CPU. Caveats honored: benchmarks are English-clip and speed-only — M0 re-measures on Chinese audio.

### 2.1 Track-first dataflow
- Headset mode (default): mic track = "You" **by source** (no ASR, no diarization); system track → VAD → ASR → diarization.
- Room-mic mode (explicit opt-in): both tracks full pipeline; 2× throughput budget; mic voiceprint pins "You".
- All timing on **per-track canonical sample indices**; cross-track echo dedup before merge; overlap renders "A + B".

### 2.2 Inference & backpressure
- Single-flight inference per engine; bounded queue in audio-seconds; chunk audio spooled to disk **before** dispatch; monotonic seq; responses idempotent by `(meeting, track, seq)`; lag → "catching up" UI, never dropped speech.
- Model loads before recording starts, never mid-meeting.

### 2.3 Attribution
- Live text/labels provisional. Finalize: full re-decode → merge with diarization turns by timestamp overlap (VAD-segment granularity). Attribution quality checked in M3; if mid-segment speaker changes matter, add diarization-informed re-segmentation (fallback design, not built by default).
- User edits during live stored as patches rebased onto final; never overwritten.

### 2.4 Text contract
Raw engine output canonical; display normalization (full/half-width punctuation, dates, phones, Chinese numerals) separate and reversible. Vocabulary hints honored where the engine supports hotwords; capability gap vs cloud models documented in UI.

### 2.5 System-audio scope
VAD transcribes any system audio (videos, notifications). Mitigations: prominent system-track mute, playback warning, meeting-app allowlist hints, source metadata. ScreenCaptureKit filtering = later.

## 3. Contingencies — decision matrix (v3.1, Codex round 2)

Single-trigger Plan B is replaced by a decision matrix. Every branch: numeric threshold, minimum hardware (target Mac spec), corpus/version pinned, measured ≥3 runs, decision recorded with owner+date.

| Failure class (M0/M1 measurement) | Action |
|---|---|
| STT capacity/latency: sustained system-track p95 caption latency > SLO or thermal RTF drift >20% | Evaluate Plan B sidecar (MPS) |
| STT quality: zh CER delta vs whisper > agreed threshold on eval set | Try alternative quant/model variant; else do not ship Qwen as default |
| sherpa API/stability/packaging: native crash, model-load failure, packaging incompatibility | Keep existing provider; defer Qwen entirely (do NOT switch STT architecture) |
| Diarization quality/license failure | Defer M3 independently; STT architecture unchanged |
| MLX arm | **Only if** a Plan A STT gate above fails (no unconditional bonus benchmark) |

## 3A. sherpa-onnx Rust API constraints (design rules)

- **Concurrency**: Rust wrappers mark objects `Sync` via unsafe declarations but expose `&self`-mutating ops (`accept_waveform`, `front/pop/flush/reset`, decode). Rule: **one recognizer owned by a dedicated inference actor, all decode serialized**; **one VAD instance per ordered track**; never concurrent calls on a shared handle. Concurrent start/stop/teardown stress tests in M0/M2.
- **Warm-start contract**: provider states `Loading → Ready → Failed`; model loads before dispatch, stays resident for the meeting, durable spool during loading; UI distinguishes "model warming" from backlog. M1 measures cold-load p50/p95, first-decode latency, warm-decode latency, peak RSS, repeated session cycles.
- **Native failure containment**: Rust constructors return `Option` and decode returns no status — empty result is ambiguous (silence/truncation/failure). Add artifact preflight (SHA + load check), structured error classification, decode-duration watchdog, crash-loop counter that disables the provider after N failures. **Decision: accept whole-app-restart on hard native fault** (no worker process in v1); durable spool makes recovery lossless — recovery tests with corrupt/missing artifacts and forced termination required in M2.
- **Generation limits**: upstream Rust defaults `max_total_len=512` / `max_new_tokens=128` are too small for fast Mandarin / number-heavy / code-switch 10s segments. Set explicit reviewed limits in M1; detect token-saturation; on saturation split the segment and retry, never silently truncate.
- **Finalize/diarization**: Rust diarization exposes a single blocking `process(samples)` (no progress callback). Finalize = bounded background job, abandonable but not interruptible; shutdown behavior explicit; UI shows indeterminate progress. M0 measures 30/60/120-min recordings, not only short clips.

## 4. Milestones

**M0 — Confirmation gate (on target M2 Air 16GB, Chinese audio, user's own recordings)**
1. **Thermal soak (highest priority — fanless machine):** 30–60 min replayed dual-track meeting; sustained RTF drift, clock throttling curve, p95/p99 caption latency, spool depth, peak RSS.
2. CER eval: 10 real meeting clips (zh, code-switch, noise, accents) vs fork's whisper path.
3. **Diarization quality gate** (not just wall time): run with `num_clusters=-1` (no oracle); across 2–6 speakers, overlap, room noise, far-field — gate on DER/JER or agreed metric, speaker-count error, boundary accuracy. Pin all artifacts `{origin, revision, SHA-256, dim, license}`.
4. License manifest complete (pyannote gating, sherpa conversion provenance, Qwen Apache-2.0).
5. **Packaged-build smoke test** (pulled forward from M4): pin sherpa crate + native libs + ONNX Runtime ABI + compile features + model artifacts as one tested compatibility set; signed release-build launch/load/decode smoke on target.
6. **Gate:** composite benchmark within SLO AND acceptable CER delta AND diarization quality gate pass → proceed. Else decision matrix (§3).

**M1 — STT provider, import path (D5 reorder; with M2 = FIRST USABLE MILESTONE)**
sherpa-onnx Rust API behind versioned `TranscriptionRequest`; warm-start contract implemented (§3A); explicit generation limits + saturation-split-retry; model manager (download/pin/verify); **import-audio-file → zh transcript** flow; golden-audio tests (zh + code-switch). No Whisper in engine surface (D8).

**M2 — Diarization + finalize for recordings (with M1 = FIRST USABLE RELEASE)**
Diarization integration (blocking `process` as bounded background job per §3A); timestamp merge; **attributed zh transcript + summary of imported recordings — user's own recordings work end-to-end (release gate)**; voiceprint profile fingerprint schema laid down (embedding-space key per Codex round 2); old Rust diarizer flag retired only after Mandarin spot-check passes; attribution quality check (§2.3).

**M3 — Live capture (dual-track)**
Track-first queues (single inference actor, serialized decode per §3A), spool, backpressure, lag UI; VAD phrase loop with deadline flush; char-alignment stitching; **room-mic mode in v1 scope (D3) + mic-voiceprint "You" pinning**; provisional→final reconcile with user locks; recovery drills (crash, kill, sleep/wake, engine switch mid-recording, corrupt artifact, forced native termination).

**M4 — Polish, CJK qualification (personal scope per D1)**
IME composition, cursor stability during segment replacement, Han-Latin spacing, full-width search, grapheme offsets, CJK fonts in PDF/DOCX; failure drills complete; **Gatekeeper bypass doc + personal build script (no signing/notarization/SBOM)**; privacy copy "local by default, offline-capable" with summary-endpoint boundary disclosed; voiceprint enrollment consent/deletion.

## 5. Risks

| Risk | Mitigation |
|---|---|
| int8 zh RTF worse than English prior | M0 gate; contingencies pre-designed |
| VAD-granularity attribution too coarse | M3 check → diarization-informed segmentation |
| Hotwords weak in sherpa path | Documented gap; Plan B escape |
| sherpa-onnx model/binary drift | Version-pin; re-run eval on bump |
| Fanless M2 Air thermals under sustained int8 inference | M0 soak is gate #1; LAN summary endpoint (D2) keeps memory free; contingencies if RTF collapses |
| pyannote gated upstream / third-party ONNX conversions | M0 manifest; personal use unaffected; redistribution decision after manifest |

## 6. Open Items
- M0 measurements (composite dual-track SLO numbers, CER, DER/JER, artifact hashes/dims) — appended when run.
- Embedding artifact final choice (`eres2net_zh-cn` vs `campplus` zh) — M0.
- Room-mic mode v1 vs v1.1 — after M0 composite benchmark.
- Sherpa Rust binding enhancement (diarization progress callback) — tracked as upstream request; not a v1 dependency.

## 7. Review history
- Round 1 (Codex plan-mode, 2026-09-01): 20 findings → verified against sources, folded as v1→v2 amendments (CAM++ 192-d, dual-track dataflow, alignment stage, milestone reorder; Codex's 256-d voiceprint claim corrected to 128-d).
- Round 2 (Codex plan-mode, 2026-09-01, verdict CONDITIONAL-GO): 11 findings → all folded as v3.1 amendments (§3 decision matrix, §3A Rust API constraints, M0 composite benchmark + diarization quality gate + packaged smoke test, voiceprint embedding-space fingerprint keys, explicit generation limits, MLX demoted to triggered-only).
