# NeuroAdapter 阶段性复现汇报

更新日期：2026-07-06

## 一句话结论

当前已经完成 NeuroAdapter 的本地复现链路打通：代码、数据、训练、checkpoint 保存、解码、brain encoder 候选选择、实验日志和 GitHub 同步都已经跑通。  
但当前结果还不能说达到论文效果，只能说完成了 subject 1 的小规模复现实验和 sanity check。

## 当前服务器项目位置

```text
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026
├── code/NeuroAdapter       # 作者原始代码
├── repro                   # 当前复现实验仓库，进 Git
├── data                    # NSD 与中间数据，不进 Git
├── outputs                 # 训练与解码结果，不进 Git
├── checkpoints             # 预留权重目录
├── tools                   # whole_brain_encoder、torch_hub 等工具
└── logs                    # 日志
```

空间占用概况：
- `data`：约 111G
- `outputs`：约 15G
- `tools`：约 1.1G
- `checkpoints`：当前为空，实际训练 checkpoint 在 `outputs/neuroadapter` 下

## 已完成内容

1. 建立复现实验仓库并同步到 GitHub private repo。
2. 下载并整理 NeuroAdapter 代码、NSD subject 1 所需数据、parcel labels、whole_brain_encoder 和相关权重。
3. 修复/适配了作者代码在当前服务器环境下的若干路径与导入问题。
4. 跑通了 subject 1 的训练链路。
5. 跑通了从 checkpoint 解码图像的流程。
6. 跑通了使用 whole_brain_encoder 对多个候选图打分并选择 best candidate 的流程。
7. 已记录每次实验结果到 `EXPERIMENT_LOG.md`。

## 已完成训练

当前最重要的训练结果已经更新为 100000 step：

```text
run: 20260707-topk100-bs4-ddp4-resume50000-to100000
initial step: 50000
final step: 100000
GPU: 4 卡 DDP，CUDA_VISIBLE_DEVICES=0,1,2,3
elapsed: 58091.15 秒，约 16.14 小时
last loss: 0.0454588011
final checkpoint:
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/outputs/neuroadapter/20260707-topk100-bs4-ddp4-resume50000-to100000/checkpoint-step-100000.pt
```

上一阶段 50000 step 训练结果：

```text
run: 20260706-topk100-bs4-ddp4-resume20000-to50000
initial step: 20000
final step: 50000
GPU: 4 卡 DDP
elapsed: 33465.75 秒，约 9.30 小时
last loss: 0.1411519349
final checkpoint:
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/outputs/neuroadapter/20260706-topk100-bs4-ddp4-resume20000-to50000/checkpoint-step-50000.pt
```

上一阶段 20000 step 训练结果：

```text
run: 20260706-topk100-bs4-ddp2-resume10000-to20000
initial step: 10000
final step: 20000
GPU: 2 卡 DDP
elapsed: 10973.29 秒，约 3.05 小时
last loss: 0.1865352988
final checkpoint:
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/outputs/neuroadapter/20260706-topk100-bs4-ddp2-resume10000-to20000/checkpoint-step-20000.pt
```

已经保留的 checkpoint 覆盖从早期 smoke test 到 100000 step，没有丢失历史权重。100000 step 训练新增保存了 `checkpoint-step-60000.pt`、`checkpoint-step-70000.pt`、`checkpoint-step-80000.pt`、`checkpoint-step-90000.pt` 和 `checkpoint-step-100000.pt`。

## 已完成解码评估

### 8 样本评估

```text
run: 20260706-steps20000-be-select8-cand8-denoise50
samples: 8
candidates per sample: 8
denoising steps: 50
positive best score: 7 / 8
```

### 50 样本评估

```text
run: 20260706-steps20000-be-select50-cand8-denoise50
samples: 50
candidates per sample: 8
denoising steps: 50
elapsed: 898.29 秒，约 14.97 分钟
positive best score: 49 / 50
positive rate: 0.98
mean best score: 0.2534
min best score: -0.0015
max best score: 0.4880
```

完整输出目录：

```text
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/outputs/neuroadapter_decode/20260706-steps20000-be-select50-cand8-denoise50
```

## 怎么理解当前结果

从工程角度看：链路已经跑通，而且 20000 step 的 brain encoder selection score 比更早 checkpoint 更稳定。

从科研复现角度看：还不能说复现出论文效果。原因是：
- 当前只是 subject 1 的小规模训练和小规模解码。
- 当前评价主要是用 whole_brain_encoder 在 8 个候选图中选 best candidate，不等同于论文完整指标。
- 人眼看图时，很多重建仍然不准确，只是在部分类别或场景上有相关性，例如冲浪/运动、飞机、鸟、室内/卫浴等。
- 生成结果仍然明显受 Stable Diffusion 先验影响，经常生成和 GT 不同的物体或场景。

因此更准确的表述是：

> 我已经把 subject 1 的训练、解码和候选选择评估流程跑通了；目前 20000 step 的 50 sample 结果是 49/50 positive、mean best score 0.2534。随后用 4 张 A40 继续训练到 50000 step，但同设置 50 sample 解码下降到 39/50 positive、mean best score 0.1664。继续训练到 100000 step 后，同设置解码恢复到 41/50 positive、mean best score 0.2172，但仍低于 20000 step。

## 为什么现在不优先继续盲目训练

继续训练当然可以，但现在继续训练的风险是：如果当前数据处理、训练配置、解码流程或评价方式和论文作者存在偏差，单纯增加 step 可能只会消耗 GPU，而不会解决视觉结果不准的问题。

当前更合理的判断是：
1. 先把已经完成的复现进展整理清楚。
2. 核对作者 README 中的默认训练/推理流程和我们当前简化脚本的差异。
3. 再决定是继续训练到更高 step，还是优先补 full evaluation / 官方 metric。

## 当前和作者默认流程的主要差异

作者 README 默认训练方式：

```bash
accelerate launch --config_file acc_config.yaml train_brain_adapter.py \
    --learning_rate 1e-04 \
    --num_train_epochs 100 \
    --train_batch_size 8 \
    --subject_id 1 \
    --topk 100 \
    --condition_dim 768 \
    --num_decoder_queries 50 \
    --sub_approach linear_projection \
    --wandb
```

当前复现实验使用的是自写的 `scripts/train_limited.py`：
- 目的是绕过环境与路径问题，先跑通 subject 1 的有限 step 训练。
- 已支持 resume 和多卡 DDP。
- 使用参数和作者默认核心参数接近：`subject 1`、`topk 100`、`condition_dim 768`、`num_decoder_queries 50`、`linear_projection`、`lr 1e-4`。
- 但不是直接跑作者原始的 `accelerate launch train_brain_adapter.py --num_train_epochs 100` 完整流程。

当前解码使用的是自写的 `scripts/decode_brain_encoder_select.py`：
- 能从 checkpoint 生成多个候选图。
- 能用 whole_brain_encoder 预测候选图对应脑响应，并和真实 fMRI 做相关，选出 best candidate。
- 这是有用的 sanity check，但还不是完整官方 `metric_brain_adapter.py` 指标。

## 当前诊断结论

已新增 `DIAGNOSIS.md`，专门记录为什么 20000 step 的 brain encoder selection 指标反而高于 50000 / 100000 step。

核心结论：
- 20000 step：49/50 positive，mean best score 0.2534
- 50000 step：39/50 positive，mean best score 0.1664
- 100000 step：41/50 positive，mean best score 0.2172
- 轻量图像指标 pixel/SSIM 与 brain encoder 指标趋势不一致，说明不能只看单一指标。
- 2026-07-09 已补上作者官方 `metric_brain_adapter.py` 评估。官方 AlexNet / Inception / CLIP 等深度特征指标整体支持 100000 step 最好。
- 当前没有发现 saved GT 与 test dataset index 错位，前 10 个样本检查 mismatch count 均为 0。
- 最可疑的问题是 resume 没有恢复 optimizer state、4 卡阶段 global batch size 变化但学习率不变，以及 brain encoder selection 与官方图像指标衡量对象不同。

官方 metric 汇总：

| checkpoint | PixCorr ↑ | SSIM ↑ | Alex(2) ↑ | Alex(5) ↑ | Incep ↑ | CLIP ↑ | Eff ↓ | SwAV ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20000 | 0.0016 | 0.2430 | 61.80 | 68.82 | 62.12 | 62.20 | 0.9395 | 0.6429 |
| 50000 | 0.0677 | 0.3084 | 68.20 | 85.84 | 78.04 | 80.69 | 0.8330 | 0.5083 |
| 100000 | 0.0757 | 0.2974 | 77.06 | 89.18 | 85.71 | 87.31 | 0.7879 | 0.4592 |

因此当前技术建议是：先做固定 seed 对照 / 配置排查，不要继续盲目长训。

## 我对下一步的判断

优先级最高的是写清楚阶段性结果，并向师兄确认下一阶段目标：

1. 如果目标是“证明代码能跑”：现在已经基本完成。
2. 如果目标是“拿到更像论文的图”：50000 step 的同设置解码指标低于 20000 step，继续训练是否有效并不确定；如果继续训练，应把它视为探索，并同步核对配置。
3. 如果目标是“严谨复现论文指标”：官方 metric 已经跑通，但还需要固定 seed 对照、更多样本/subject 和训练配置消融。

我的建议路线：

```text
短期：
先汇报当前结果，说明链路已跑通，结果还不是论文级。

中期：
做固定 seed 解码和训练配置对照，确认 100000 step 在官方图像指标上的优势是否稳定。

长期：
50000 step 的同设置解码已经完成，指标低于 20000 step。后续如果继续长训，应在训练完成后立即做同设置解码，并重点排查 global batch size、学习率、数据对应和评价指标。
```

## 如果师兄问“跑通了吗”

可以回答：

> 跑通了 subject 1 的主要链路：数据加载、训练、checkpoint、解码、brain encoder 候选选择评估和作者官方 metric 都能跑。现在已经训练到 100000 step；brain encoder selection 上 20000 step 最高，但官方 AlexNet / Inception / CLIP 指标上 100000 step 最高。

## 如果师兄问“下一步打算干什么”

可以回答：

> 下一步技术上不应该继续盲目加训练步数，因为不同指标趋势不一致。应该先做固定 seed 对照和配置排查，尤其是 global batch size、学习率、resume 训练动态，以及 brain encoder selection 和官方图像指标之间的差异。
