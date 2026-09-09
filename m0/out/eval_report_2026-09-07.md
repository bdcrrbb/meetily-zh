# M0 Spike 评测报告 — STT (CER) + Speaker Diarization

日期：2026-09-07
分支：`m0-spike`（commit 048d7d1）
评测人：bing（评测执行由 Claude Code 协助完成）

---

## 1. 概要

| 项目 | 结果 | 判定 |
|---|---|---|
| STT 转写质量（POOLED CER，人工 refs） | **0.2497**（数字归一后 0.2424） | ✅ 通过（难音频上符合 0.6B int8 预期） |
| STT 速度（RTF，30s clip，3 线程） | 0.20–0.30（正常段），0.61（远场难段） | ✅ 通过 |
| Diarization，短音频（10 min，th=0.6） | 7 speakers，分布健康（59%/24%/11%） | ⚠️ 可用，需调 threshold |
| Diarization，长音频（64 min） | th=0.5→104 / 0.6→76 / 0.7→58 speakers | ❌ **不通过**，碎裂随时长累积，单点 threshold 无法修复 |
| Diarization fixed-k 路径 | k=4 强制聚类退化：98.1% 归单 speaker | ❌ 不可用（与 threshold 路径行为矛盾） |

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
- 10 个 clip，15–45s：

| clip | 起点 | 时长 | 特点 |
|---|---|---|---|
| clip01 | 600s | 30s | 清晰普通话 |
| clip02 | 1200s | 45s | 连续长段 |
| clip03 | 1800s | 30s | 普通话为主 |
| clip04 | 2400s | 40s | 清晰段（最好成绩） |
| clip05 | 3000s | 30s | 中英混杂（"翻墙"、"豆包"、"AI"） |
| clip06 | 2100s | 30s | 独立时间戳（原 -ss 600 与 clip01 重复，已重切） |
| clip07 | 1500s | 35s | 补切 |
| clip08 | 2700s | 30s | 噪声开头 + 非语音段（最难 clip） |
| clip09 | 3300s | 30s | 远场难段 |
| clip10 | 3600s | 30s | 远场难段 |

---

## 4. STT 评测

### 4.1 方法

两轮评测：

1. **Bootstrap 轮**：refs 由独立模型家族 whisper large-v3-turbo（mlx-whisper，本地推理）自动转写生成。目的：sanity check。**已知缺陷**：whisper 在此音频上同样出错（幻觉，如 clip08 "facit不了"、clip01 "投资"→"投掷AI"），测得 CER 0.3397 为两模型分歧度，**不是** Qwen3 真实错误率。
2. **人工轮（最终）**：评测人逐 clip 听写 refs（`m0/out/refs.json`），重跑 `m0 cer`。以下数字以此为准。

### 4.2 结果（人工 refs）

```
POOLED CER: 0.2497 (222/889)          ← m0 cer 原生输出
数字归一后: 0.2424 (216/891)           ← 额外折算中文数字 vs 阿拉伯数字
```

Per-clip（升序）：

| clip | CER | | clip | CER |
|---|---|---|---|---|
| clip04 | **0.107** | | clip02 | 0.259 |
| clip09 | 0.143 | | clip01 | 0.262 |
| clip10 | 0.232 | | clip05 | 0.278 |
| clip03 | 0.239 | | clip07 | 0.292 |
| clip06 | 0.244 | | clip08 | **0.678** |

要点：

- **9/10 clip 落在 0.11–0.29**，唯一离群是 clip08（见 4.3）。此评测集是刻意难例（远场+口语+噪声），近场麦会议录音预期显著更好。
- Bootstrap refs 轮的 per-clip 数字与人工轮排名大洗牌（例：clip09 whisper-ref 0.307 → 人工 ref 0.143 — 该段 m0 听得比 whisper 好）。教训：**bootstrap CER 只能当 pooled sanity check，per-clip 与 gate 决策必须人工 ref**。
- CER 0.25 中相当比例是语气词/量词级替换（"他的人也进去了" vs "他的人进去了"），对下游 LLM summary 语义保真影响有限。建议 M1 增加语义保留度指标，不要只盯 CER。

### 4.3 clip08（CER 0.678）解剖 — 唯一真实质量问题

```
HYP: 那题目里，可转化里，就是我们的那个。我希望我们现在用这个来讲我们未来的。我要活一百五十年……
REF:                                    我希望我们现在用这个来养活我们未来的。我要活150年……
```

- **开头 hallucination**：在噪声/换气 lead-in 上凭空生成 13 字（"那题目里，可转化里，就是我们的那个。"）。这是 decode 阶段对非语音输入缺乏抑制，产品侧建议：VAD lead-in trim / 首段置信度过滤（VAD 已部分覆盖，但此段 VAD 放行了）。
- 中段替换："寿命不会短的" → "什么也不会管"（远场音质导致）。

### 4.4 STT 速度

| 指标 | 值 |
|---|---|
| clip RTF（10/10） | 0.199–0.296（8 个 clip） |
| 远场难段 RTF（clip09/10） | 0.602 / 0.614 |
| 30s clip decode 中位耗时 | ~6–9s（难段 ~18s） |

远场难段 RTF 恶化 2 倍，real-time 相机仍安全（RTF < 1），但 soak 测试应包含此类音频。

---

## 5. Diarization 评测

### 5.1 方法

- 全片：`evalset/meeting_full.wav`（3844.8s，16k mono）
- 为控制单次 ~11–14 min 的运行成本，先在**前 10 min**（`/tmp/meeting_head10.wav`）做 threshold / num-clusters sweep，再将胜出配置跑全片验证。
- 评测维度：speaker 数量、talk-time 分布、turn 结构。

### 5.2 Sweep 结果

**head-10min（600s）：**

| 配置 | speakers | talk-time 分布 | 评价 |
|---|---|---|---|
| auto（th=0.5） | 10 | 41%/17%/16%/11%/9%/4% | 偏碎 |
| th=0.4 | 22 | 平坦长尾 | 过碎 |
| th=0.3 | 46 | 平坦长尾 | 严重过碎 |
| th=0.6 | **7** | **59%/24%/11%/5%** | **最健康**：1 主讲 + 1 主要参与者 + 长尾 |
| th=0.7 | 6 | 70%/24%/5% | 开始过合并 |
| fixed k=2…6 | 2/2/3/3/4 | **单 speaker 99–100%** | 全部退化，不可用 |

**全片（3844.8s）：**

| 配置 | speakers | talk-time 分布 |
|---|---|---|
| auto（th=0.5） | 104 | 平坦：top 仅 12.4%，73/104 speaker <10s |
| th=0.6 | 76 | top-2 = 53%（28%/25%），74 个长尾 speaker |
| th=0.7 | 58 | 未解决 |
| fixed k=4 | 4 | **退化：spk0 98.1%，其余 3 个仅 2%** |

### 5.3 发现

1. **`threshold` 语义未在 CLI 文档说明**。实测为 cosine **distance** 阈值：调高 = 合并更激进 = speaker 更少（0.3→46、0.4→22、0.5→10、0.6→7、0.7→6，单调）。建议写入 `--help`。
2. **fixed-k 路径与 threshold 路径行为矛盾，此音频上不可用**。强制 k=4 得到 98% 单块 + 碎屑；而 threshold 路径明明能分出 59%/24% 双主体。sherpa-onnx `FastClusteringConfig` 两条代码路径（num_clusters>0 vs -1）表现不一致，直接用会得到完全错误的结构。
3. **核心失败模式：embedding 相似度贴地 + greedy 聚类错误累积**。eres2net 在此远场压缩音频上，same-speaker 相似度紧贴合并阈值：短音频选好 threshold（0.6）即可得到健康结构；但每个 segment 的合并判断是局部 greedy，一次"该并未并"即永久分裂，64 min 的错误累积机会是 10 min 的 6 倍+ → speaker 数从 7 恶化到 76。**这不是 threshold 调参问题，是算法结构问题**——0.5/0.6/0.7 全片均失败。
4. **数据侧的真实说话人数推断**：th=0.6 head 分布（59%/24%/11%）强烈暗示该音频为 1 主讲 + 1–2 主要参与者 + 偶发插话。最终需人耳确认（评测人待办）。

### 5.4 对产品路径的影响与建议

64 min 直接喂 `m0 diarize` 不可用。按优先级：

1. **分层 two-pass 聚类（推荐）**：每 ~10 min 窗口内独立聚类（此尺度上 th=0.6 已验证可用）→ 用各窗口 speaker centroid 做第二层合并，跨窗口对齐同一说话人。把 greedy 错误局部化在窗口内，第二层只处理少量、较长的 centroid，噪声小。
2. **评估联合模型**：当前 pipeline 用 pyannote-segmentation-3.0 只做切分、再拼独立 eres2net embedding。pyannote 3.0 自身输出 speaker activation，可做联合打分或与 embedding 互验，绕开单靠 embedding 相似度的贴地问题。
3. **产品 UX 兜底**：会议录音场景 speaker 数通常已知（2–8 人），可让用户指定人数走 fixed-k——但前提是先修复发现 2（fixed-k 路径退化），否则不可用。
4. **评测侧**：正式 speaker diarization 指标建议引入 DER/JER（需要人工标注或至少抽样对齐），当前 talk-time 分布只能定性。

---

## 6. 问题清单

| # | 严重度 | 问题 | 建议 |
|---|---|---|---|
| P1 | 高 | 长音频（>30min）diarization 碎裂，speaker 数随时长线性膨胀（64min→58–104 个） | 分层 two-pass 聚类（见 5.4.1） |
| P1 | 高 | fixed-k 聚类路径退化（k=4 → 98% 单 speaker），与 threshold 路径矛盾 | 排查 sherpa FastClustering 两路径差异；修复前勿对外暴露 `--num-clusters` |
| P2 | 中 | 非语音 lead-in 上 ASR hallucinate 前缀（clip08：13 字） | VAD lead-in trim / 首段置信度过滤 |
| P2 | 中 | `cer` normalization 不折算中文数字 vs 阿拉伯数字（"一百五十" vs "150" 记 6 错，pooled 影响 ~0.7pt） | cer 归一化加数字折算 |
| P3 | 低 | `diarize --threshold` 语义（cosine distance，越高越合并）未文档化 | 补 `--help` 说明 |
| P3 | 低 | 远场难段 RTF 恶化 2 倍（0.30→0.61） | soak 用例加入远场音频 |

---

## 7. 复现命令

```bash
# 评测集（示例）
ffmpeg -y -ss 600 -t 30 -i evalset/audio.mp4 -ar 16000 -ac 1 evalset/clip01.wav

# STT（10 clips）
./m0/target/release/m0 stt --wavs evalset/clip01.wav ... evalset/clip10.wav \
    --out m0/out/stt_corpus.json

# CER（refs = 人工听写）
python3 -c "..."   # stt_corpus.json -> hyps.json ({clip: text})
./m0/target/release/m0 cer --refs m0/out/refs.json --hyps m0/out/hyps.json --out m0/out/cer.json

# Diarization sweep（head-10min）
ffmpeg -y -t 600 -i evalset/meeting_full.wav -ar 16000 -ac 1 /tmp/meeting_head10.wav
for th in 0.3 0.4 0.5 0.6 0.7; do
  ./m0/target/release/m0 diarize --wav /tmp/meeting_head10.wav \
      --num-clusters=-1 --threshold=$th --out /tmp/diar_head_th$th.json
done
# 全片验证
./m0/target/release/m0 diarize --wav evalset/meeting_full.wav \
    --num-clusters=-1 --threshold=0.6 --out m0/out/diar_full_th06.json
```

## 8. 产物文件清单

| 文件 | 内容 |
|---|---|
| `m0/out/stt_corpus.json` | 10 clip 转写 + RTF（m0） |
| `m0/out/refs.json` | 人工听写 references |
| `m0/out/hyps.json` | m0 hypotheses（{clip: text}） |
| `m0/out/cer.json` | CER 逐 clip + pooled（人工 refs 版，最终） |
| `m0/out/diar_full_k4.json` | 全片 fixed k=4（退化证据） |
| `m0/out/diar_full_th06.json` | 全片 th=0.6（76 spk） |
| `m0/out/diar_full_th07.json` | 全片 th=0.7（58 spk） |
| `m0/out/diar_full.json` | 全片 auto th=0.5（104 spk，原始失败样本） |
| `/tmp/diar_head_th*.json`, `/tmp/diar_head_k*.json` | head-10min sweep 中间产物 |

## 9. 待办（评测人）

- [ ] 人耳确认 meeting 真实 speaker 数（数据推断：1 主讲 + 1–2 主要参与者）
- [ ] 决定是否将分层聚类列入 M1 scope
- [ ] 近场麦克风录音重跑同套评测（预期 CER 显著低于 0.25）
