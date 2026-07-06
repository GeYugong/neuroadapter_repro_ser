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

当前最重要的训练结果已经更新为 50000 step：

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

已经保留的 checkpoint 覆盖从早期 smoke test 到 50000 step，没有丢失历史权重。50000 step 训练新增保存了 `checkpoint-step-25000.pt`、`checkpoint-step-30000.pt`、`checkpoint-step-35000.pt`、`checkpoint-step-40000.pt`、`checkpoint-step-45000.pt` 和 `checkpoint-step-50000.pt`。

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

> 我已经把 subject 1 的训练、解码和候选选择评估流程跑通了；目前 20000 step 的小规模结果在 brain encoder 分数上有正相关趋势，随后又用 4 张 A40 继续训练到了 50000 step。50000 step 的解码评估还没跑，视觉效果是否改善需要下一步验证。

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

## 我对下一步的判断

优先级最高的是写清楚阶段性结果，并向师兄确认下一阶段目标：

1. 如果目标是“证明代码能跑”：现在已经基本完成。
2. 如果目标是“拿到更像论文的图”：下一步应先用 `checkpoint-step-50000.pt` 跑和 20000 step 相同的解码评估，再判断是否继续训练到完整 epoch。
3. 如果目标是“严谨复现论文指标”：下一步应补官方 full evaluation / metric，而不是只看小样本图。

我的建议路线：

```text
短期：
先汇报当前结果，说明链路已跑通，结果还不是论文级。

中期：
补一个更正式的 evaluation/metric，确认当前结果在数字指标上处于什么水平。

长期：
先用 50000 step checkpoint 跑同设置解码评估，再决定是否继续训练到更长。
```

## 如果师兄问“跑通了吗”

可以回答：

> 跑通了 subject 1 的主要链路：数据加载、训练、checkpoint、解码和 brain encoder 候选选择评估都能跑。现在已经训练到 50000 step；20000 step 已做了 50 个测试样本、每个 8 个候选图的评估，49/50 的 best candidate brain encoder score 为正。50000 step 的解码还没做，下一步要用同样设置对比 20000 和 50000。

## 如果师兄问“下一步打算干什么”

可以回答：

> 我觉得下一步不应该继续盲目加训练步数。现在已经到 50000 step，应先用 50000 checkpoint 跑同样的 50 sample 解码，和 20000 step 的结果做对比。如果 50000 视觉效果明显改善，再考虑继续；如果没有改善，就要优先查配置和评价流程。
