#!/usr/bin/env bash
# M0 model/artifact downloads (pinned URLs). Records SHA-256 for docs/model-manifest.json.
set -uo pipefail
DIR="${1:-models}"
mkdir -p "$DIR"
cd "$DIR"

dl() { # url, retries with resume
  local url="$1" out="$2"
  for i in 1 2 3 4 5 6 7 8; do
    curl -L --max-time 900 -C - -o "$out" "$url" && return 0
    echo "retry $i for $out"; sleep 3
  done
  return 1
}

echo "== silero_vad.onnx"
dl https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx silero_vad.onnx

echo "== qwen3-asr-0.6B-int8 (~1.9GB, resumable)"
dl https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2 qwen3.tar.bz2
tar xf qwen3.tar.bz2 && rm qwen3.tar.bz2

echo "== pyannote segmentation-3.0"
dl https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 pyannote.tar.bz2
tar xf pyannote.tar.bz2 && rm pyannote.tar.bz2

echo "== 3D-Speaker eres2net zh-cn embedding"
dl https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

echo "== 3D-Speaker campplus zh (alt embedding, M0 comparison)"
dl https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx || echo "campplus zh optional - continuing"

echo "== SHA-256 manifest =="
sha256sum silero_vad.onnx 3dspeaker_*.onnx sherpa-onnx-pyannote-segmentation-3-0/*.onnx sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/*.onnx 2>/dev/null | tee sha256.txt
echo "DONE"
