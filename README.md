# NeuroAdapter 功能脑区因果研究

这个仓库最初用于 NeuroAdapter 论文复现，目前的主要研究目标是：

> 研究不同功能性 fMRI 脑区对类别特异性图像重建的因果贡献，并分析
> top-SNR parcel selection 是否会对脑区贡献结论产生偏差。

Appendix P 的严格复现与已有 50-sample zero-mask 结果被保留为
legacy/exploratory，不作为最终功能脑区结论。新研究采用公开的
Algonauts Project 2023 Subject 1 fsaverage ROI masks。

研究计划、协议与当前状态见：

```text
docs/RESEARCH_PLAN.md
docs/EXPERIMENT_PROTOCOL.md
docs/CURRENT_STATE.md
docs/DECISIONS.md
```

## 当前阶段结果

阶段 A/B 已完成，尚未启动 E2 扩散消融或新模型训练。

- parcel 干预位置已修正为 `ParcelMapper` 之后、`TokenMapper` 之前；
- 服务器单元测试结果为 `15 passed`；
- E0 全 1000 parcel inventory 显示，公开映射得到的 97 个功能 ROI
  parcel 全部进入 top-SNR-200，因此没有观察到 Face、Word、V4
  覆盖不足；
- E1 最终选择 37 张 Face、50 张 Body、50 张 Scene 确认性刺激；
- 24 张 Word 候选中有 21 张不与确认性集合重叠；因缺少 OCR 证据，
  仍只保留为 exploratory。

详细证据、图表、限制和下一步门槛见 `docs/CURRENT_STATE.md`。

## 推荐项目入口

长期工作时优先从这个目录进入本项目：

```text
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026
├── code/NeuroAdapter       # 作者原始 NeuroAdapter 代码
├── repro                   # 当前复现实验 Git 仓库
├── data                    # NSD 和转换后的大数据，不进 Git
├── outputs                 # 训练与解码输出，不进 Git
├── checkpoints             # 权重目录，不进 Git
├── tools
│   ├── whole_brain_encoder # brain encoder 外部工具
│   └── torch_hub           # DINOv2 torch hub 本地代码
└── logs                    # 日志目录，不进 Git
```

VS Code 推荐打开：

```text
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/repro
```

## 兼容路径

旧路径仍保留为 symlink，用于兼容历史脚本中的绝对路径：

```text
/public/home/mty/GeYugong/code/NeuroAdapter -> projects/neuroadapter-iclr2026/code/NeuroAdapter
/public/home/mty/GeYugong/neuroadapter-repro -> projects/neuroadapter-iclr2026/repro
/public/home/mty/GeYugong/data -> projects/neuroadapter-iclr2026/data
/public/home/mty/GeYugong/outputs -> projects/neuroadapter-iclr2026/outputs
/public/home/mty/GeYugong/checkpoints -> projects/neuroadapter-iclr2026/checkpoints
/public/home/mty/GeYugong/logs -> projects/neuroadapter-iclr2026/logs
/public/home/mty/GeYugong/tools/whole_brain_encoder -> projects/neuroadapter-iclr2026/tools/whole_brain_encoder
/public/home/mty/GeYugong/tools/torch_hub -> projects/neuroadapter-iclr2026/tools/torch_hub
```

## 已验证环境

```bash
conda activate neuroadapter
export PYTHONNOUSERSITE=1
export PYTHONPATH=/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/code/NeuroAdapter
```

兼容旧路径也可用：

```bash
export PYTHONPATH=/public/home/mty/GeYugong/code/NeuroAdapter
```

已跑通核心 fake-data 测试：

```bash
cd /public/home/mty/GeYugong/projects/neuroadapter-iclr2026/code/NeuroAdapter
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
/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/code/NeuroAdapter
external/NeuroAdapter -> /public/home/mty/GeYugong/code/NeuroAdapter
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

## 新代码路径约定

新代码通过以下环境变量或 CLI 参数解析路径，不再增加服务器绝对路径：

```text
NEUROADAPTER_PROJECT_ROOT
NEUROADAPTER_UPSTREAM_ROOT
NEUROADAPTER_DATA_ROOT
NEUROADAPTER_OUTPUT_ROOT
NEUROADAPTER_CHECKPOINT_ROOT
```

示例配置见 `configs/paths.example.yaml`。
