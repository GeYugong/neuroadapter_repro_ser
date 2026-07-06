# NeuroAdapter 复现实验仓库

这个仓库用于记录 NeuroAdapter 论文复现过程中的配置、脚本、笔记和实验说明。

## 推荐项目入口

长期工作时优先从这个目录进入本项目：

```text
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026
├── code        -> ../../code/NeuroAdapter
├── repro       -> ../../neuroadapter-repro
├── data        -> ../../data
├── outputs     -> ../../outputs
├── checkpoints -> ../../checkpoints
├── tools       -> ../../tools
└── logs        -> ../../logs
```

当前这些条目是 symlink。原因是早期脚本和正在运行/已经完成的实验使用了 `/public/home/mty/GeYugong/...` 下的绝对路径；直接移动真实目录会破坏训练、checkpoint、decode 记录。

VS Code 推荐打开：

```text
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/repro
```

## 兼容路径

这些是真实目录或既有入口，当前仍保留：

```text
/public/home/mty/GeYugong
├── code/NeuroAdapter       # 作者原始 NeuroAdapter 代码
├── neuroadapter-repro      # 当前复现实验 Git 仓库
├── data                    # NSD 和转换后的大数据，不进 Git
├── outputs                 # 训练与解码输出，不进 Git
├── checkpoints             # 权重目录，不进 Git
├── tools                   # whole_brain_encoder、DINOv2 等外部工具
└── logs                    # 日志目录，不进 Git
```

后续如果要彻底整理，可以在所有训练停止后，把真实目录迁移进 `projects/neuroadapter-iclr2026/`，并在旧路径保留 symlink 兼容历史脚本。

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

作者源码入口：

```text
external/NeuroAdapter -> /public/home/mty/GeYugong/code/NeuroAdapter
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/code -> /public/home/mty/GeYugong/code/NeuroAdapter
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
