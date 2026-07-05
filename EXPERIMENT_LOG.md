# NeuroAdapter 复现实验记录

本文件记录服务器 `/public/home/mty/GeYugong` 上 NeuroAdapter 复现的完整过程。大数据、模型权重和训练输出不进入 Git，只记录路径、配置、结果和问题。

## 环境与代码

- 服务器账号：`mty`
- 工作区：`/public/home/mty/GeYugong`
- 作者代码：`/public/home/mty/GeYugong/code/NeuroAdapter`
- 复现仓库：`/public/home/mty/GeYugong/neuroadapter-repro`
- Conda 环境：`neuroadapter`
- GPU：6 x NVIDIA A40，每张约 46GB 显存
- 已确认 fake model 测试可运行：`brain_adapter/model.py`

## 2026-07-05 数据准备

### NSD subject 1 下载

脚本：`scripts/download_nsd_subj01.sh`

原始数据位置：

```text
/public/home/mty/GeYugong/data/nsd
```

下载内容：

- `nsd_stimuli.hdf5`
- `nsd_expdesign.mat`
- `nsd_stim_info_merged.csv`
- `nsd_stim_info_merged.pkl`
- `subj01/fsaverage/betas_fithrf_GLMdenoise_RR` 下左右脑 40 个 session 的 `.mgh`

结果：下载完成，原始数据约 `74G`。

### NeuroAdapter 数据格式转换

脚本：`scripts/prepare_nsd_subj01.py`

输出位置：

```text
/public/home/mty/GeYugong/data/neuroadapter/neural_data
```

生成文件：

- `metadata_sub-01.npy`
- `betas_sub-01.h5`

结果：转换完成，NeuroAdapter neural data 约 `37G`。

### Schaefer parcel 标签

脚本：`scripts/generate_schaefer_labels.py`

来源：ThomasYeoLab/CBIG 的 `Schaefer2018_1000Parcels_7Networks_order.annot`，FreeSurfer `fsaverage`。

输出位置：

```text
/public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer
```

生成文件：

- `lh_labels_s01.pt`
- `rh_labels_s01.pt`

验证结果：

```text
lh: 501 个标签，覆盖 163842 个顶点
rh: 501 个标签，覆盖 163842 个顶点
```

其中第 0 个标签是 medial wall，NeuroAdapter 的 dataset 会跳过它，所以每个半球有效 parcel 数为 500。

## 2026-07-05 数据读取验证

测试内容：真实 NSD subject 1 dataloader。

配置：

- split: `test`
- topk: `100`
- subject: `1`

结果：

```text
test set 长度: 1000
num_parcels: 200
max_voxels: 626
img_encoder:   (3, 425, 425)
img_ipadapter: (3, 512, 512)
brain_lh_f:    (100, 626)
brain_rh_f:    (100, 626)
```

结论：数据读取链路可用。

## 2026-07-05 Smoke Train

脚本：`scripts/smoke_train_step.py`

目的：验证真实数据能否完成一次训练步，不代表论文复现指标。

配置：

- subject: `1`
- topk: `4`
- batch size: `1`
- mixed precision: `no`
- GPU: `cuda`

结果：

```text
smoke_train_step ok
loss 0.053140923380851746
num_parcels 8
max_voxels 440
img_ipadapter (1, 3, 512, 512)
brain_lh_f (1, 4, 440)
brain_rh_f (1, 4, 440)
device cuda
```

问题与处理：

- 作者代码 `train_brain_adapter.py` 导入了 `nsd_groupwise_topk_parcel_dataset`，但当前 `dataset.py` 中没有该符号。smoke 脚本中为单被试训练做了兼容补丁。
- 服务器存在失效代理 `127.0.0.1:17891`，运行 Hugging Face/Diffusers 相关命令时需要 unset proxy。
- `fp16` 下遇到 `Attempting to unscale FP16 gradients`，smoke 测试改用 FP32，即 `mixed_precision=no`。

结论：真实 NSD 数据 -> dataloader -> Stable Diffusion -> NeuroAdapter -> loss -> backward -> optimizer step 已跑通。

## 2026-07-05 Limited Train Run 1

脚本：`scripts/train_limited.py`

计划配置：

- subject: `1`
- topk: `100`
- batch size: `1`
- max steps: `50`
- mixed precision: `no`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- 输出目录：`/public/home/mty/GeYugong/outputs/neuroadapter/<run-name>`

状态：已完成。

运行结果：

```text
run name: 20260705-topk100-bs1-steps50
dataset_len: 9000
num_parcels: 200
max_voxels: 626
device: cuda
torch: 2.4.1+cu121
耗时: 58.41 秒
first_loss: 0.049765344709157944
last_loss: 0.0046898601576685905
min_loss: 0.0026548015885055065
max_loss: 0.4162288308143616
```

输出目录：

```text
/public/home/mty/GeYugong/outputs/neuroadapter/20260705-topk100-bs1-steps50
```

生成文件：

```text
config.json
losses.csv
summary.json
checkpoint-step-0025.pt
checkpoint-step-0050.pt
```

checkpoint：

```text
/public/home/mty/GeYugong/outputs/neuroadapter/20260705-topk100-bs1-steps50/checkpoint-step-0050.pt
```

是否 OOM/报错：无。

结论：`topk=100`、`batch_size=1` 的真实训练可以跑通；显存约 8.3GB，速度约 1 step/s。50 step 只是链路和稳定性验证，不代表论文指标。

## 2026-07-05 Limited Train Run 2

脚本：`scripts/train_limited.py`

配置：

- subject: `1`
- topk: `100`
- batch size: `1`
- max steps: `500`
- mixed precision: `no`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- save every: `100`

运行结果：

```text
run name: 20260705-topk100-bs1-steps500
dataset_len: 9000
num_parcels: 200
max_voxels: 626
device: cuda
torch: 2.4.1+cu121
耗时: 422.43 秒
first_loss: 0.049765344709157944
last_loss: 0.08804396539926529
min_loss: 0.00226954510435462
max_loss: 0.4880872070789337
显存: 约 8.5GB
``` 

输出目录：

```text
/public/home/mty/GeYugong/outputs/neuroadapter/20260705-topk100-bs1-steps500
```

生成文件：

```text
config.json
losses.csv
summary.json
checkpoint-step-0100.pt
checkpoint-step-0200.pt
checkpoint-step-0300.pt
checkpoint-step-0400.pt
checkpoint-step-0500.pt
```

最终 checkpoint：

```text
/public/home/mty/GeYugong/outputs/neuroadapter/20260705-topk100-bs1-steps500/checkpoint-step-0500.pt
```

是否 OOM/报错：无。

结论：`topk=100`、`batch_size=1` 训练 500 steps 稳定完成，速度约 `1.18 step/s`。loss 有波动，500 steps 仍然只是小规模稳定性验证，不代表论文复现效果。

## 2026-07-05 Decode Smoke Test 1

脚本：`scripts/decode_limited.py`

目的：验证训练出的 checkpoint 能否加载，并从 test fMRI 生成图片。此实验不做 brain encoder 候选排序，也不代表论文指标。

配置：

- checkpoint: `/public/home/mty/GeYugong/outputs/neuroadapter/20260705-topk100-bs1-steps500/checkpoint-step-0500.pt`
- checkpoint step: `500`
- subject: `1`
- topk: `100`
- test samples: `4`
- start index: `0`
- denoising steps: `20`
- noise factor: `4.0`
- num predictions: `1`
- mixed precision: `fp16`
- GPU: `CUDA_VISIBLE_DEVICES=0`

运行结果：

```text
run name: 20260705-steps500-decode4
耗时: 27.42 秒
输出 grid: /public/home/mty/GeYugong/outputs/neuroadapter_decode/20260705-steps500-decode4/grid_gt_pred.png
```

输出目录：

```text
/public/home/mty/GeYugong/outputs/neuroadapter_decode/20260705-steps500-decode4
```

生成文件：

```text
sample_0000_gt.png / sample_0000_pred.png / sample_0000_gt_pred.png
sample_0001_gt.png / sample_0001_pred.png / sample_0001_gt_pred.png
sample_0002_gt.png / sample_0002_pred.png / sample_0002_gt_pred.png
sample_0003_gt.png / sample_0003_pred.png / sample_0003_gt_pred.png
grid_gt_pred.png
summary.json
```

结果观察：生成流程成功，图片文件非空，拼图尺寸为 `512x1024`。右列预测图目前与左列 ground truth 没有明显语义对应，这符合预期，因为模型只训练了 500 steps，当前目标是打通 decode 链路而非得到论文级效果。

![Decode smoke test: ground truth left, prediction right](assets/20260705-steps500-decode4-grid_gt_pred.png)

结论：checkpoint 加载、test fMRI 输入、Stable Diffusion 生成、图片保存链路已经跑通。

## 2026-07-05 Batch Size 4 Stability Test

脚本：`scripts/train_limited.py`

目的：测试 `batch_size=4` 是否能在单张 A40 上稳定训练，为后续更长训练选择 batch size。

配置：

- subject: `1`
- topk: `100`
- batch size: `4`
- max steps: `50`
- mixed precision: `no`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- save every: `50`

运行结果：

```text
run name: 20260705-topk100-bs4-steps50
dataset_len: 9000
num_parcels: 200
max_voxels: 626
device: cuda
torch: 2.4.1+cu121
耗时: 150.67 秒
first_loss: 0.15584427118301392
last_loss: 0.16647395491600037
min_loss: 0.015826208516955376
max_loss: 0.31141236424446106
显存: 约 16.5GB
```

输出目录：

```text
/public/home/mty/GeYugong/outputs/neuroadapter/20260705-topk100-bs4-steps50
```

结论：`batch_size=4` 不会 OOM，单步约 3 秒，样本吞吐略高于 `batch_size=1`。后续长训练可以使用 `batch_size=4`，但需要支持从已有 checkpoint 继续训练，避免重复从零开始。
