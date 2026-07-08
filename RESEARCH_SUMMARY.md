# NeuroAdapter 复现研究总结报告

更新日期：2026-07-08

## 1. 当前结论

目前已经完成 NeuroAdapter 在服务器上的第一轮复现工作：项目目录、代码、数据、训练、checkpoint、解码、候选图生成、brain encoder 选择、轻量指标诊断、实验记录和 GitHub 同步都已经建立起来。

但当前结果还不能表述为“复现出论文效果”。更准确的说法是：

> 已经跑通 subject 1 的主要工程链路，并完成到 100000 step 的探索性训练；当前生成图能看到一定类别相关性，但仍不是稳定、精确的 brain-to-image 重建。不同评价指标之间也不一致，因此下一步应优先补官方 metric 和训练配置对照，而不是继续盲目长训。

## 2. 项目与目录状态

服务器项目根目录：

```text
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026
├── code/NeuroAdapter       # 作者原始代码
├── repro                   # 当前复现实验 Git 仓库
├── data                    # NSD 与转换后的数据，不进 Git
├── outputs                 # 训练与解码结果，不进 Git
├── checkpoints             # 预留权重目录，不进 Git
├── tools                   # whole_brain_encoder、DINOv2 torch hub 等
└── logs                    # 日志
```

VS Code 推荐打开：

```text
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/repro
```

复现仓库 GitHub remote：

```text
git@github.com:GeYugong/neuroadapter-repro.git
```

## 3. 已完成工作

### 3.1 资料与仓库准备

已整理 NeuroAdapter 论文复现所需核心材料，包括：

- 项目主页、arXiv、PDF、HTML、OpenReview。
- 作者代码仓库 `kriegeskorte-lab/NeuroAdapter`。
- `whole_brain_encoder` 及相关 brain encoder 权重。
- NSD 数据集、AWS Open Data、Deep Image Reconstruction 等背景数据链接。
- IP-Adapter、Stable Diffusion v1.5 等相关背景材料。

已建立私有复现仓库，用来保存：

- 实验脚本。
- 配置说明。
- 研究笔记。
- 实验日志。
- 小型结果图和摘要 JSON。

大数据、checkpoint、完整生成图不进入 Git。

### 3.2 数据准备

已完成 subject 1 所需数据准备：

- 下载 NSD subject 1 相关数据。
- 转换 NeuroAdapter 需要的 neural data 格式。
- 生成 Schaefer parcel 标签。
- 验证 train/test 数据读取链路。

关键验证结果：

```text
test set length: 1000
num_parcels: 200
max_voxels: 626
img_encoder:   (3, 425, 425)
img_ipadapter: (3, 512, 512)
brain_lh_f:    (100, 626)
brain_rh_f:    (100, 626)
```

目前没有发现图像 trial 顺序、train/test 划分或 parcel 对应关系的明显错误。

### 3.3 代码适配

作者代码不能在当前服务器环境中无修改直接完整跑通，所以为了先完成链路验证，补充了若干复现实验脚本：

- `scripts/train_limited.py`：有限 step 训练脚本，支持 resume 和 DDP。
- `scripts/decode_limited.py`：基础解码 smoke test。
- `scripts/decode_brain_encoder_select.py`：生成多个候选图，并用 whole_brain_encoder 选择 best candidate。
- `scripts/summarize_decode_runs.py`：汇总多个解码 run 的 brain encoder selection 指标。
- `scripts/evaluate_decode_outputs.py`：对 PNG 输出计算 pixel correlation 和 SSIM。
- `scripts/check_data_alignment.py`：检查 saved GT 和 dataset index 是否一致。

这些脚本的目的不是替代论文作者完整流程，而是先让复现可以落地、可记录、可诊断。

### 3.4 训练链路

已经跑通：

- fake model 测试。
- 真实 NSD dataloader。
- 单步 smoke training。
- 单卡小规模训练。
- 单卡 resume 训练。
- 2 卡 DDP 训练。
- 4 卡 DDP 训练。

当前最终训练到：

```text
checkpoint-step-100000.pt
```

路径：

```text
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/outputs/neuroadapter/20260707-topk100-bs4-ddp4-resume50000-to100000/checkpoint-step-100000.pt
```

### 3.5 解码与候选选择

已经跑通：

- 从 checkpoint 加载 NeuroAdapter。
- 用 test fMRI 生成图像。
- 每个样本生成多个候选图。
- 使用 whole_brain_encoder 对候选图预测脑响应。
- 与真实 fMRI parcel response 计算相关。
- 从候选图中选择 best candidate。

当前主对比采用统一设置：

```text
samples: 50
candidates per sample: 8
denoising steps: 50
topk: 100
selection metric: whole_brain_encoder dinov2_q enc_1 run_1 lh/rh mean score
```

## 4. 主要训练结果

| 阶段 | run | GPU | 起止 step | 用时 | last loss |
|---|---|---:|---:|---:|---:|
| 早期长训 | `20260705-topk100-bs4-resume2250-to10000` | 1 A40 | 2250 -> 10000 | 2.47 h | 0.1144 |
| 20k | `20260706-topk100-bs4-ddp2-resume10000-to20000` | 2 A40 | 10000 -> 20000 | 3.05 h | 0.1865 |
| 50k | `20260706-topk100-bs4-ddp4-resume20000-to50000` | 4 A40 | 20000 -> 50000 | 9.30 h | 0.1412 |
| 100k | `20260707-topk100-bs4-ddp4-resume50000-to100000` | 4 A40 | 50000 -> 100000 | 16.14 h | 0.0455 |

注意：20k 之后训练条件发生了变化。20k 阶段是 2 卡，每卡 batch size 4，global batch size 约 8；50k 和 100k 阶段是 4 卡，每卡 batch size 4，global batch size 约 16。学习率仍保持 `1e-4`。

## 5. 主要解码结果

### 5.1 Brain Encoder Selection 指标

| checkpoint | positive best score | positive rate | mean best score | min | max |
|---:|---:|---:|---:|---:|---:|
| 20000 | 49 / 50 | 0.98 | 0.2534 | -0.0015 | 0.4880 |
| 50000 | 39 / 50 | 0.78 | 0.1664 | -0.2202 | 0.5139 |
| 100000 | 41 / 50 | 0.82 | 0.2172 | -0.2131 | 0.5294 |

从这个指标看：

- 20000 step 最高。
- 50000 step 明显下降。
- 100000 step 相比 50000 step 有恢复，但仍低于 20000 step。

这就是之前说“后面没有稳定提升”的依据。

### 5.2 轻量图像指标

| checkpoint | brain encoder mean | pixel corr mean | SSIM mean |
|---:|---:|---:|---:|
| 20000 | 0.2534 | 0.0016 | 0.2277 |
| 50000 | 0.1664 | 0.0678 | 0.2945 |
| 100000 | 0.2172 | 0.0758 | 0.2827 |

这个结果很重要：不同指标的趋势并不一致。

- 按 brain encoder selection：`20000 > 100000 > 50000`
- 按 pixel correlation：`100000 > 50000 > 20000`
- 按 SSIM：`50000 > 100000 > 20000`
- 按肉眼观感：100000 可能更自然、更像正常图像

所以不能简单说“100000 一定更差”。更严谨的表述是：

> 100000 step 的视觉自然性可能更好，但当前 brain encoder selection 指标没有超过 20000 step。评价标准之间存在分歧，需要进一步跑官方完整 metric。

## 6. 目前对图像效果的判断

从预览图看，模型已经不是纯随机输出。部分样本会出现与 GT 大类相关的内容，例如：

- 冲浪、海边、运动人物。
- 食物、厨房、室内场景。
- 猫、鸟、飞机等常见类别。

但问题仍然明显：

- 很多图像只是在大类或氛围上相关，不是精确重建。
- 物体类别、数量、位置经常错。
- 生成结果仍明显受 Stable Diffusion 先验影响。
- brain encoder 分数较高不一定代表人眼语义最像。
- 人眼觉得更自然的图，也不一定 brain encoder 分数最高。

因此当前结果可以作为“链路跑通和阶段性探索结果”，不能作为“论文效果已经复现”的证据。

## 7. 已完成诊断

### 7.1 数据对应关系

已用 `scripts/check_data_alignment.py` 检查三组 50 sample 解码的前 10 个样本：

```text
20000 mismatch_count: 0
50000 mismatch_count: 0
100000 mismatch_count: 0
```

结论：当前没有证据表明 saved GT 图和 test dataset index 错位。

### 7.2 训练 resume 风险

当前 `train_limited.py` 的 checkpoint 主要保存模型权重和 losses，没有完整保存 optimizer / scheduler state。每次 resume 时，AdamW optimizer 会重新创建。

这意味着：

- 20000 -> 50000 不是严格意义上的完整连续训练。
- 50000 -> 100000 也不是严格意义上的完整连续训练。
- AdamW 的动量、一阶矩、二阶矩等状态没有恢复。

这可能影响继续训练后的效果。

### 7.3 多卡与 batch size 风险

20k 之后从 2 卡换到 4 卡：

```text
20k 阶段：2 张卡 x 每卡 batch 4 = global batch 8
50k/100k 阶段：4 张卡 x 每卡 batch 4 = global batch 16
```

但学习率仍然是：

```text
1e-4
```

因此，后面效果不稳定有可能与 global batch size 改变有关。现在不能把 20k、50k、100k 简单理解成“同一设置下只增加 step”的公平对比。

### 7.4 官方 metric 还没跑通

作者的 `metric_brain_adapter.py` 需要的格式和当前输出不同：

- 作者 metric 需要 `evaluation_metadata.json`。
- 作者 metric 需要 `sample_summary.json`。
- 作者 metric 需要 `sample_000001.npz` 形式的样本文件。
- 当前输出是 `summary.json + gt.png + candidate_*.png`。

依赖检查结果：

```text
clip missing
skimage ok
torchvision ok
pandas ok
scipy ok
```

因此目前还没有完成论文正式指标评估。

## 8. 当前工作产物

### 8.1 文档

- `README.md`：仓库和目录说明。
- `EXPERIMENT_LOG.md`：完整实验流水账。
- `REPORT.md`：阶段性复现汇报。
- `DIAGNOSIS.md`：为什么长训指标不稳定的技术诊断。
- `RESEARCH_SUMMARY.md`：阶段性研究总结报告。

### 8.2 诊断文件

```text
diagnostics/decode_brain_encoder_summary.json
diagnostics/decode_lightweight_image_metrics.json
diagnostics/data_alignment_20000_first10.json
diagnostics/data_alignment_50000_first10.json
diagnostics/data_alignment_100000_first10.json
```

### 8.3 关键图片预览

```text
assets/20260706-steps20000-be-select50-cand8-denoise50-preview12.png
assets/20260707-steps50000-be-select50-cand8-denoise50-preview12.png
assets/20260708-steps100000-be-select50-cand8-denoise50-preview12.png
```

完整大图保存在服务器 `outputs/neuroadapter_decode` 下，不进入 Git。

## 9. 生成图视觉检查

本节把当前仓库 `assets/` 中保存的所有生成图都放进报告，并逐张记录肉眼观察。这里的判断是视觉检查，不等同于正式论文指标。

### 9.1 500 step 基础解码

![500 step decode smoke](assets/20260705-steps500-decode4-grid_gt_pred.png)

观察：

- 左列是 GT，右列是模型生成。
- 右列基本没有和 GT 建立语义对应。
- 菜市场生成成类似贝壳/物体，厨房生成成模糊帽子状物体，冲浪和帆板也没有对应。
- 结论：这张图只证明最早期 checkpoint 能加载、能生成、能保存图片，不代表有效重建。

### 9.2 2250 step 基础解码，20 denoising steps

![2250 step decode smoke](assets/20260705-steps2250-decode4-grid_gt_pred.png)

观察：

- 生成图大多是灰色背景或单个孤立物体。
- 有一张出现天空/海面/鸟的组合，但与厨房 GT 不相关。
- 相比 500 step 没有明显语义提升。
- 结论：2250 step 仍然远远不够，基础单候选解码不可用。

### 9.3 2250 step 基础解码，50 denoising steps

![2250 step decode 50 denoising steps](assets/20260705-steps2250-decode4-denoise50-grid_gt_pred.png)

观察：

- denoising steps 从 20 增到 50 后，生成图更像正常 Stable Diffusion 图片。
- 但语义仍不对：市场图生成成猫/人物，厨房生成成儿童/人物，冲浪生成成湖面，帆板生成成直升机。
- 结论：采样步数增加能改善图片自然性，但不能解决 brain-to-image 对齐问题。

### 9.4 2250 step，4 候选 brain encoder selection

![2250 step brain encoder candidate selection](assets/20260705-steps2250-be-select4-cand4-denoise20-grid.png)

观察：

- 这是第一次接入候选图选择：每个 GT 后面有多个 candidate，红框是 brain encoder 选出的 best candidate。
- 四个样本的 best score 仍然都是负数。
- 被选中的图和 GT 基本不匹配：市场对应灰图，厨房对应黑白儿童照，冲浪对应人物合影，帆板对应剪影。
- 结论：候选选择流程跑通了，但 2250 step 模型本身还没有学到可靠映射。

### 9.5 10000 step，4 候选 brain encoder selection

![10000 step 4-candidate selection](assets/20260706-steps10000-be-select4-cand4-denoise20-grid.png)

观察：

- 图像自然性比 2250 step 明显提高。
- 候选中开始出现海边、运动人物、飞机等有用先验。
- 但四个样本 best score 仍然为负，红框候选仍然不可靠。
- 结论：10000 step 已经比早期更像正常图，但还不能说明有效重建。

### 9.6 10000 step，8 候选，50 denoising steps

![10000 step 8-candidate selection](assets/20260706-steps10000-be-select8-cand8-denoise50-grid.png)

观察：

- 候选数量和采样步数增加后，图像质量继续提高。
- 一些样本开始有正分，例如食物、室外、运动相关候选。
- 但大多数 best candidate 仍然和 GT 差距很大，例如市场被选成室内房间，厨房被选成文字牌，食物被选成书本/窗户，猫被选成狗。
- 结论：这张图证明“多候选生成 + brain encoder selection”可运行，但还只是 sanity check。

### 9.7 20000 step，8 候选，50 denoising steps

![20000 step 8-candidate selection](assets/20260706-steps20000-be-select8-cand8-denoise50-grid.png)

观察：

- 20000 step 比 10000 step 更稳定，8 个样本里 7 个 best score 为正。
- 部分类别出现相关性，例如冲浪样本候选里出现运动/海边元素，帆板样本候选里出现水面/天空/户外元素。
- 但红框 best candidate 仍常常不是人眼最像 GT 的图；例如菜市场被选成建筑/室内，厨房被选成人像。
- 结论：20000 step 在 brain encoder selection 指标上明显优于 10000 step，但视觉重建仍不精确。

### 9.8 20000 step，50 样本预览

![20000 step 50-sample preview](assets/20260706-steps20000-be-select50-cand8-denoise50-preview12.png)

观察：

- 这是 50 sample 大评估的前 12 个样本预览。
- 从指标看这一组最高：49/50 positive，mean best score 0.2534。
- 视觉上能看到一些类别相关候选：冲浪、飞机、猫、运动/球拍等。
- 但市场、厨房、食物、咖啡等样本经常被选成室内、人物、鸟、飞机或其他错类。
- 结论：20000 step 的脑表征选择分数最高，但肉眼看并不是最自然的一组。

### 9.9 50000 step，50 样本预览

![50000 step 50-sample preview](assets/20260707-steps50000-be-select50-cand8-denoise50-preview12.png)

观察：

- 这组图肉眼看比 20000 step 更自然，尤其冲浪、食物、猫、飞机等类别明显更像正常图片。
- 冲浪样本基本都生成了冲浪/滑雪/海浪相关图像；猫样本也生成了多张猫；飞机样本多张都是飞机。
- 但 brain encoder selection 指标下降到 39/50 positive，mean best score 0.1664。
- 这说明图像自然性和当前 brain encoder selection 分数不完全一致。
- 结论：不能简单说 50000 step 更差；更准确是“brain encoder 指标变差，但肉眼图像自然性可能变好”。

### 9.10 100000 step，50 样本预览

![100000 step 50-sample preview](assets/20260708-steps100000-be-select50-cand8-denoise50-preview12.png)

观察：

- 这是目前肉眼最像正常图像的一组。
- 厨房/室内、冲浪、披萨/食物、猫、飞机等类别都更成型。
- 但仍存在明显错配：菜市场变成餐厅，食物被稳定生成成披萨，咖啡变成植物/钟表/蔬菜，部分运动样本被生成成网球或滑雪。
- 指标上 100000 step 比 50000 step 恢复，但仍低于 20000 step：41/50 positive，mean best score 0.2172。
- 结论：100000 step 肉眼效果可能最好，但 brain encoder selection 不是最高。当前必须区分“视觉自然性”和“脑表征匹配分数”。

## 10. 目前还没有完成的事

以下内容还没完成，不能在汇报中说已经完成：

1. 没有跑作者原版完整 `accelerate launch train_brain_adapter.py --num_train_epochs 100` 流程。
2. 没有跑官方完整 `metric_brain_adapter.py`。
3. 没有完成所有 subject 的训练与评估。
4. 没有下载或处理完整 NSD 全量配置。
5. 没有证明结果达到论文表格或论文图示水平。
6. 没有做固定 seed 的公平解码对照。
7. 没有做 2 卡 vs 4 卡、global batch 8 vs 16 的严格消融。

## 11. 后续工作计划

下一阶段不宜直接继续长训。主要原因是 100000 step 的 brain encoder selection 指标仍未超过 20000 step，且当前还没有完成官方正式 metric。后续工作应优先围绕评价流程和训练配置排查展开。

1. **补官方 metric 格式转换**
   把当前 PNG 解码输出转换为作者 metric 所需的 `.npz + metadata` 结构。

2. **补齐 metric 依赖**
   安装或确认 `clip`，确认 deep feature 相关权重缓存，然后跑官方完整指标。

3. **做固定 seed 解码**
   对 20k、50k、100k 使用相同样本、相同候选数、相同 denoising steps、可复现 seed，减少扩散随机性的干扰。

4. **做训练配置小对照**
   从 20k checkpoint 出发，保持 global batch size 不变，或降低学习率到 `1e-5`，只训练 5000-10000 step，再看指标是否改善。

5. **再决定是否继续长训**
   如果官方 metric 和固定 seed 对照显示 100k 确实更好，再考虑继续训练。否则继续长训可能只是消耗 GPU 时间。

## 12. 阶段性汇报摘要

本阶段已跑通 NeuroAdapter 在 subject 1 上的主要工程链路，包括 NSD 数据准备、parcel 标签生成、训练、checkpoint 保存、解码、多候选生成、whole_brain_encoder 候选选择和实验日志记录。目前已训练到 100000 step。

50 sample 的 brain encoder selection 结果如下：

| checkpoint | positive best score | mean best score |
|---:|---:|---:|
| 20000 | 49 / 50 | 0.2534 |
| 50000 | 39 / 50 | 0.1664 |
| 100000 | 41 / 50 | 0.2172 |

从视觉结果看，100000 step 的生成图自然性较好，厨房/室内、冲浪、食物、猫、飞机等类别更成型。但从 brain encoder selection 指标看，100000 step 仍未超过 20000 step。当前结果说明复现链路已经跑通，但还不能证明达到论文效果。

后续重点应放在三个方面：补官方完整 metric，做固定 seed 的公平解码对照，排查 2 卡到 4 卡后 global batch size 改变以及 optimizer state 未恢复对训练结果的影响。
