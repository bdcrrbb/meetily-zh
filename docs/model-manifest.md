# Model Artifact Manifest — M0
Generated: 2026-09-03 (sha256s from m0/models/sha256.txt on server; harness re-verifies on use)

| Artifact | Origin | SHA-256 | License / Terms | Notes |
|---|---|---|---|---|
| sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 (conv_frontend/encoder/decoder/tokenizer) | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2 | conv_frontend: d22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e; decoder: 4f6885be5959ae26af3089d38ee7972c5fafbeeb1cf8d5e76eab6d8b61ca5771; encoder: 60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9 | Qwen3-ASR weights Apache-2.0; ONNX conversion by Wasser1462/Qwen3-ASR-onnx, packaged by k2-fsa | int8; zh + 30 langs + 22 zh dialects |
| silero_vad.onnx | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx | 9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6 | MIT (silero-vad) | |
| sherpa-onnx-pyannote-segmentation-3-0/model.onnx | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | 220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079 | pyannote segmentation model: MIT-licensed code; upstream HF weights gated (accepted terms required upstream) — this is k2-fsa's converted copy | see also model.int8.onnx d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d |
| 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx | 1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b | Apache-2.0 (3D-Speaker, Alibaba) | zh-cn trained; dim to be asserted at runtime (192-d class) |
| 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx | f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11 | Apache-2.0 (3D-Speaker) | alternate embedding; M0 picks one |
| sherpa-onnx Rust crate | crates.io, version 1.13.7 (Cargo.lock pinned) | — | Apache-2.0/MIT | downloads prebuilt static libs at build time (SHERPA_ONNX_ARCHIVE_DIR cacheable) |

## Gate status reference
- Capacity/thermal: PASS (docs/m0-results.md)
- CER: pending corpus
- Diarization quality: pending annotated clips
