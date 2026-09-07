# M0 Spike 评测报告 — STT (CER) + Speaker Diarization

日期：2026-09-07
分支：`m0-spike`（commit 048d7d1）
评测人：bing（评测执行由 Claude Code 协助完成）

---

## 1. 概要

| 项目                                  | 结果                                  | 判定                                                  |
| ------------------------------------- | ------------------------------------- | ----------------------------------------------------- |
| STT 转写质量（POOLED CER，人工 refs） | **0.2497**（数字归一后 0.2424）       | ✅ 通过（难音频上符合 0.6B int8 预期）                 |
| STT 速度（RTF，30s clip，3 线程）     | 0.20–0.30（正常段），0.61（远场难段） | ✅ 通过                                                |
| Diarization，短音频（10 min，th=0.6） | 7 speakers，分布健康（59%/24%/11%）   | ⚠️ 可用，需调 threshold                                |
| Diarization，长音频（64 min）         | th=0.5→104 / 0.6→76 / 0.7→58 speakers | ❌ **不通过**，碎裂随时长累积，单点 threshold 无法修复 |
| Diarization fixed-k 路径              | k=4 强制聚类退化：98.1% 归单 speaker  | ❌ 不可用（与 threshold 路径行为矛盾）                 |

**核心结论**：Qwen3-ASR-0.6B int8 转写质量达标，可作为 product path。**Diarization 是本 spike 的主要负面结果**：eres2net embedding 在远场压缩音频上 same-speaker 相似度贴近阈值，greedy 聚类错误随音频时长线性累积，64 min 会议直接喂 pipeline 会产生几十个虚假 speaker。产品化需要分层聚类（分窗→合并）或换联合模型。

---

## 2. 测试环境

- 硬件/系统：Apple Silicon Mac，macOS (Darwin 25)
- 二进制：`m0/target/release/m0`（commit 048d7d1，含 10s force-split）
- 模型（`models/`）：
  - ASR：`sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`
  - VAD：`silero_vad.onnx`
  - 分段：`sherpa-onnx-pyannote-segmentation-3-0`
  - Speaker embedding：`3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`
- ONNX Runtime 线程：3（默认）

---

## 3. 评测集

- 源音频：`evalset/audio.mp4`，64 min（3840s）中文多人对话视频，**刻意选的难例**：远场收音、口语重、背景噪声、压缩音轨。
- 采样：`ffmpeg -ar 16000 -ac 1` 转 16k 单声道 wav。
- 10 个 clip，15–45s（clip01–clip10，起点 600s–3600s，覆盖清晰普通话 / 中英混杂 / 噪声开头 / 远场难段）。

---

## 4. STT 评测

### 4.1 方法

1. **Bootstrap 轮**：refs 由 whisper large-v3-turbo (mlx-whisper) 自动生成 → CER 0.3397 为两模型分歧度，非真实错误率。
2. **人工轮（最终）**：评测人逐 clip 听写 refs，重跑 `m0 cer`。

### 4.2 结果（人工 refs）

```
POOLED CER: 0.2497 (222/889)
数字归一后: 0.2424 (216/891)
```

Per-clip：clip04 0.107（最好）/ clip09 0.143 / clip10 0.232 / clip03 0.239 / clip06 0.244 / clip02 0.259 / clip01 0.262 / clip05 0.278 / clip07 0.292 / clip08 0.678（唯一离群）。

要点：
- 9/10 clip 落在 0.11–0.29；评测集为刻意难例，近场麦预期显著更好。
- Bootstrap refs per-clip 与人工轮排名大洗牌（clip09: whisper-ref 0.307 → 人工 ref 0.143）——**bootstrap CER 只能当 pooled sanity check，gate 决策必须人工 ref**。
- CER 中相当比例为语气词/量词级替换，对 LLM summary 语义保真影响有限。建议 M1 增加语义保留度指标。

### 4.3 clip08（CER 0.678）解剖

- **开头 hallucination**：噪声/换气 lead-in 上凭空生成 13 字。产品侧：VAD lead-in trim / 首段置信度过滤。
- 中段替换："寿命不会短的"→"什么也不会管"（远场音质）。

### 4.4 STT 速度

clip RTF 0.199–0.296（8/10），远场难段 0.602/0.614（恶化 2×，仍 <1）。soak 用例应加入远场音频。

---

## 5. Diarization 评测

### 5.1 方法

全片 meeting_full.wav（3844.8s）；先 head-10min sweep（threshold 0.3–0.7 + fixed k=2…6），胜出配置跑全片。

### 5.2 结果

head-10min：th=0.3→46 spk / 0.4→22 / 0.5→10 / **0.6→7（59%/24%/11%，最健康）** / 0.7→6；fixed k=2…6 全部退化（单 speaker 99–100%）。

全片：auto(0.5)→104 spk（73 个 <10s）/ 0.6→76 / 0.7→58 / fixed k=4 → spk0 占 98.1%。

### 5.3 发现

1. `threshold` 实测为 cosine **distance** 阈值（越高越合并），未在 CLI 文档说明。
2. **fixed-k 路径与 threshold 路径行为矛盾**（k=4 → 98% 单块 vs threshold 路径可分 59%/24% 双主体），sherpa FastClustering 两路径不一致。
3. **核心失败模式**：远场压缩音频上 embedding 相似度贴地 + greedy 合并错误随时长线性累积（10min→7 spk，64min→76 spk）。结构问题，非调参问题。
4. 数据推断真实说话人数：1 主讲 + 1–2 主要参与者 + 偶发插话（待人耳确认）。

### 5.4 建议（优先级序）

1. **分层 two-pass 聚类**：每 ~10 min 窗口独立聚类（th=0.6 已验证）→ 窗口 centroid 第二层合并对齐。
2. 评估 pyannote segmentation 自身 speaker activation 与 embedding 互验。
3. UX 兜底：用户指定人数走 fixed-k——前提是先修复发现 2。
4. 正式指标引入 DER/JER。

---

## 6. 问题清单

| # | 严重度 | 问题 | 建议 |
|---|---|---|---|
| P1 | 高 | 长音频 diarization 碎裂（64min→58–104 spk） | 分层 two-pass 聚类 |
| P1 | 高 | fixed-k 聚类路径退化 | 排查 sherpa 两路径差异；修复前不暴露 --num-clusters |
| P2 | 中 | 非语音 lead-in 上 ASR hallucinate 前缀 | VAD lead-in trim / 首段置信度过滤 |
| P2 | 中 | cer normalization 不折算中文数字 | cer 归一化加数字折算（pooled 影响 ~0.7pt） |
| P3 | 低 | diarize --threshold 语义未文档化 | 补 --help |
| P3 | 低 | 远场难段 RTF 恶化 2× | soak 用例加入远场音频 |
