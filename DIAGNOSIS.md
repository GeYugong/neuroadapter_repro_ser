# NeuroAdapter 复现问题诊断

更新日期：2026-07-09

## 当前问题

继续训练并没有让 brain encoder selection 指标稳定提升；固定 seed 对照后，这个现象仍然存在。

同一组设置：

```text
samples: 50
candidates per sample: 8
denoising steps: 50
topk: 100
selection metric: whole_brain_encoder dinov2_q enc_1 run_1 lh/rh mean score
```

结果：

| checkpoint | positive best score | mean best score | min | max |
|---:|---:|---:|---:|---:|
| 20000 | 49 / 50 | 0.2534 | -0.0015 | 0.4880 |
| 50000 | 39 / 50 | 0.1664 | -0.2202 | 0.5139 |
| 100000 | 41 / 50 | 0.2172 | -0.2131 | 0.5294 |

结论：100000 比 50000 有恢复，但仍低于 20000。因此不能继续用“训练更久就会更好”作为默认假设。

固定 seed 对照结果：

```text
seed: 12345
seed strategy: seed + dataset_idx
samples: 50
candidates per sample: 8
denoising steps: 50
topk: 100
```

| checkpoint | positive best score | mean best score | min | max |
|---:|---:|---:|---:|---:|
| 20000 | 47 / 50 | 0.2357 | -0.0067 | 0.5282 |
| 50000 | 39 / 50 | 0.1678 | -0.2086 | 0.5113 |
| 100000 | 40 / 50 | 0.2041 | -0.2172 | 0.5331 |

固定 seed 后仍是 20000 > 100000 > 50000，说明该分歧不是简单由扩散采样随机性造成的。

## 已完成的非训练排查

### 1. 训练配置核对

已检查三个主要训练 run 的 `summary.json`：

```text
20000:
run: 20260706-topk100-bs4-ddp2-resume10000-to20000
initial step: 10000
final step: 20000
num processes: 2
batch size: 4 per process
learning rate: 1e-4
optimizer state: 未恢复，只恢复模型权重

50000:
run: 20260706-topk100-bs4-ddp4-resume20000-to50000
initial step: 20000
final step: 50000
num processes: 4
batch size: 4 per process
learning rate: 1e-4
optimizer state: 未恢复，只恢复模型权重

100000:
run: 20260707-topk100-bs4-ddp4-resume50000-to100000
initial step: 50000
final step: 100000
num processes: 4
batch size: 4 per process
learning rate: 1e-4
optimizer state: 未恢复，只恢复模型权重
```

关键风险：
- 从 2 卡到 4 卡后，effective global batch size 从 8 增到 16。
- 学习率仍固定为 `1e-4`，没有随 global batch 或训练阶段调整。
- `scripts/train_limited.py` 的 checkpoint 只保存模型权重和 losses，没有保存 optimizer / scheduler state。
- 每次 resume 都是重新创建 AdamW optimizer，这和作者原版 `accelerator.save_state(...)` 的训练状态恢复不同。
- 这可能解释为什么继续长训并不单调改善。

### 2. 评价方式核对

当前主要结果来自 `scripts/decode_brain_encoder_select.py`：
- 生成多个 candidate image。
- 用 whole_brain_encoder 预测每张 candidate 对应的脑响应。
- 将预测脑响应与真实 fMRI parcel response 做 Pearson correlation。
- 选 mean correlation 最高的 candidate。

这不是作者 `metric_brain_adapter.py` 的完整图像指标。

作者 metric 脚本评估：
- Pixel correlation
- SSIM
- AlexNet two-way identification
- Inception two-way identification
- CLIP two-way identification
- EfficientNet feature similarity
- SwAV feature similarity

原始解码输出不能直接喂给作者 metric：
- 作者 metric 需要 `evaluation_metadata.json`
- 作者 metric 需要 `sample_summary.json`
- 作者 metric 需要 `sample_000001.npz` 形式的样本文件
- 当前解码输出是 `summary.json + gt.png + candidate_*.png`

已新增 `scripts/prepare_metric_inputs.py` 将当前输出转换为官方 metric 所需格式，并安装 OpenAI CLIP、缓存 Inception / EfficientNet / SwAV 权重。20000 / 50000 / 100000 三组 50 sample 结果均已跑通 `metric_brain_adapter.py`。

官方 metric 结果：

| checkpoint | PixCorr ↑ | SSIM ↑ | Alex(2) ↑ | Alex(5) ↑ | Incep ↑ | CLIP ↑ | Eff ↓ | SwAV ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20000 | 0.0016 | 0.2430 | 61.80 | 68.82 | 62.12 | 62.20 | 0.9395 | 0.6429 |
| 50000 | 0.0677 | 0.3084 | 68.20 | 85.84 | 78.04 | 80.69 | 0.8330 | 0.5083 |
| 100000 | 0.0757 | 0.2974 | 77.06 | 89.18 | 85.71 | 87.31 | 0.7879 | 0.4592 |

说明：`PixCorr`、`SSIM`、AlexNet、Inception、CLIP 越高越好。作者脚本中的 `Eff` 和 `SwAV` 实际使用 correlation distance，越低越好。

固定 seed 官方 metric 结果：

| checkpoint | PixCorr ↑ | SSIM ↑ | Alex(2) ↑ | Alex(5) ↑ | Incep ↑ | CLIP ↑ | Eff ↓ | SwAV ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20000 | 0.0360 | 0.2242 | 59.39 | 70.41 | 58.57 | 62.86 | 0.9363 | 0.6328 |
| 50000 | 0.0715 | 0.3167 | 71.14 | 84.08 | 74.33 | 85.27 | 0.8473 | 0.5112 |
| 100000 | 0.0893 | 0.3038 | 80.04 | 90.98 | 85.14 | 89.22 | 0.7864 | 0.4620 |

轻量图像指标结果：

| checkpoint | brain mean | pixel corr mean | SSIM mean |
|---:|---:|---:|---:|
| 20000 | 0.2534 | 0.0016 | 0.2277 |
| 50000 | 0.1664 | 0.0678 | 0.2945 |
| 100000 | 0.2172 | 0.0758 | 0.2827 |

这说明不同指标给出的趋势不一致：
- brain encoder selection 认为 20000 最好。
- pixel correlation / SSIM 认为 50000 或 100000 更好。
- 官方 AlexNet / Inception / CLIP 深度特征指标认为 100000 最好。

所以当前不能只靠一个指标下结论。固定 seed 后仍然是同样结论：brain encoder selection 和官方图像指标衡量的对象不同，随机 seed 不是主要解释。

### 3. 数据和索引对应关系检查

已新增 `scripts/check_data_alignment.py`，检查 `summary.json` 中记录的 `dataset_idx` 对应的 saved GT PNG 是否和 `nsd_topk_parcel_dataset(split="test")` 中同 index 图像一致。

检查范围：
- 20000 run 前 10 个样本
- 50000 run 前 10 个样本
- 100000 run 前 10 个样本

结果：

```text
20000 mismatch_count: 0
50000 mismatch_count: 0
100000 mismatch_count: 0
```

这说明至少当前检查范围内，test dataset index 与 saved GT 图像是一致的。当前没有证据表明 GT 图错位。

### 4. 视觉观察

从预览图看：
- 20000 的 brain encoder score 最高，但一些图像仍明显偏离 GT。
- 50000 的 brain encoder score 下降，但 pixel/SSIM 更高，可能更偏向低层图像相似或生成风格变化。
- 100000 相比 50000 有恢复，部分类别更稳定，例如冲浪、食物、猫、飞机，但仍不是精确重建。

视觉和指标共同说明：当前结果只能说明链路跑通，不能说明已经复现论文效果。

## 当前最可能的问题来源

按优先级排序：

1. **resume 没有恢复 optimizer state**
   - 长训阶段从 checkpoint 只加载模型权重。
   - AdamW 动量等状态每次重新开始。
   - 这可能导致 20000 -> 50000 / 50000 -> 100000 的动态和真正连续训练不同。

2. **global batch size 改变但学习率没调整**
   - 10000 -> 20000 使用 2 卡，每卡 batch 4，global batch 8。
   - 20000 后使用 4 卡，每卡 batch 4，global batch 16。
   - 学习率仍为 `1e-4`。
   - 这可能使后续训练不稳定或偏离。

3. **brain encoder selection 和官方图像指标不是同一个东西**
   - brain encoder selection 是一种候选选择 sanity check。
   - 官方 metric 是图像质量/识别指标。
   - 当前已经观察到 brain score 与官方图像指标趋势不一致。

4. **候选生成随机性已排查，不能解释主要分歧**
   - 已补固定 seed 对照。
   - 固定 seed 后 brain encoder selection 仍支持 20000，官方图像指标仍支持 100000。
   - 因此随机性会影响个别样本，但不是当前指标分歧的主因。

5. **作者原版训练流程和当前 `train_limited.py` 仍有差异**
   - 原版按 epoch 保存 `accelerator.save_state(...)`。
   - 当前按 step 保存自定义 `.pt`。
   - 当前脚本是为了先跑通复现链路而写的有限训练脚本，不是作者完整训练流程。

## 已新增脚本

### `scripts/summarize_decode_runs.py`

汇总一个或多个 `summary.json` 的 brain encoder selection 结果。

示例：

```bash
python scripts/summarize_decode_runs.py \
  /path/to/20000/summary.json \
  /path/to/50000/summary.json \
  /path/to/100000/summary.json \
  --json-out diagnostics/decode_brain_encoder_summary.json
```

### `scripts/evaluate_decode_outputs.py`

对当前 PNG 解码输出计算轻量图像指标：
- pixel correlation
- SSIM

示例：

```bash
python scripts/evaluate_decode_outputs.py \
  /path/to/20000/summary.json \
  /path/to/50000/summary.json \
  /path/to/100000/summary.json \
  --json-out diagnostics/decode_lightweight_image_metrics.json
```

### `scripts/check_data_alignment.py`

检查 saved GT PNG 是否和 NSD test dataset 对应 index 一致。

示例：

```bash
python scripts/check_data_alignment.py \
  --summary /path/to/summary.json \
  --checkpoint /path/to/checkpoint-step-100000.pt \
  --max-samples 10
```

### `scripts/prepare_metric_inputs.py`

将 `decode_brain_encoder_select.py` 输出的 `summary.json + PNG` 转换成作者 `metric_brain_adapter.py` 需要的 `.npz + metadata` 结构。

示例：

```bash
python scripts/prepare_metric_inputs.py \
  /path/to/decode/summary.json \
  --output-dir /path/to/neuroadapter_metric_inputs/run-name
```

### `scripts/summarize_official_metrics.py`

汇总多个 `metric_subset.json` 的官方 metric 结果。

示例：

```bash
python scripts/summarize_official_metrics.py \
  /path/to/20k/metric_subset.json \
  /path/to/50k/metric_subset.json \
  /path/to/100k/metric_subset.json \
  --json-out diagnostics/official_metric_summary.json
```

## 已生成诊断文件

```text
diagnostics/decode_brain_encoder_summary.json
diagnostics/decode_brain_encoder_summary_seed12345.json
diagnostics/decode_lightweight_image_metrics.json
diagnostics/official_metric_summary.json
diagnostics/official_metric_summary_seed12345.json
diagnostics/data_alignment_20000_first10.json
diagnostics/data_alignment_50000_first10.json
diagnostics/data_alignment_100000_first10.json
```

## 下一步建议

不建议继续直接长训。

更合理的下一步：

1. 做训练配置小对照，重点排查 global batch size、学习率和 optimizer state：
   - 从 20000 checkpoint 开始
   - 使用和 20000 阶段一致的 global batch size，或单独比较 global batch 8 与 16
   - 学习率降到 `1e-5` 或单独比较 `1e-4` 与 `1e-5`
   - 若继续 resume，应保存/恢复 optimizer state，或明确把实验标记为只加载模型权重的探索性训练
   - 只训练 5000-10000 step
   - 立刻跑同设置 50 sample decode
   - 同时跑 brain encoder selection 和官方 metric

当前最应该避免的是：继续用 `lr=1e-4`、4 卡、长 step 盲训，然后只看单一指标。官方图像指标支持 100000 step 更好，但 brain encoder selection 仍支持 20000 step，因此下一步应先做公平对照。
