# NeuroAdapter 复现实验仓库

这个仓库用于记录 NeuroAdapter 论文复现过程中的配置、脚本、笔记和实验说明。

## 服务器目录

```text
/public/home/mty/GeYugong
├── code/NeuroAdapter              # 原始 NeuroAdapter 代码
├── neuroadapter-repro             # 当前复现实验仓库
├── data                           # 大数据目录，不进 Git
├── checkpoints                    # 权重目录，不进 Git
└── logs                           # 日志目录，不进 Git
```

## 已验证环境

```bash
conda activate neuroadapter
export PYTHONNOUSERSITE=1
export PYTHONPATH=/public/home/mty/GeYugong/code/NeuroAdapter
```

已跑通核心 fake-data 测试：

```bash
cd /public/home/mty/GeYugong/code/NeuroAdapter
PYTHONNOUSERSITE=1 PYTHONPATH=$PWD conda run -n neuroadapter python brain_adapter/model.py
```

预期输出包含：

```text
Output fMRI features shape: torch.Size([2, 100, 768])
Output condition token shape: torch.Size([2, 50, 768])
Grad checked
```

## 当前源码位置

源码软链接：

```text
external/NeuroAdapter -> /public/home/mty/GeYugong/code/NeuroAdapter
```

## 后续需要确认

训练前需要向师兄确认这些路径：

```text
metadata_sub-01.npy
betas_sub-01.h5
nsd_stimuli.hdf5
lh_labels_s01.pt
rh_labels_s01.pt
whole_brain_encoder checkpoints
Stable Diffusion v1.5 权重或 Hugging Face 权限
```

## Git 原则

进 Git：

```text
README、配置、脚本、笔记、小型结果摘要
```

不进 Git：

```text
NSD 数据、权重、checkpoint、生成图、大日志、wandb 输出
```
