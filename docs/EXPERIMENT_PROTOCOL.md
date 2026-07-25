# 实验协议

## 配对原则

对于给定的 subject、刺激、checkpoint 和随机种子，所有实验条件必须使用
相同的 fMRI 样本、初始 VAE latent、扩散噪声、去噪调度、guidance scale
和模型权重。不同条件之间只能改变 parcel 干预。

## 干预位置

```text
fMRI beta
  -> ParcelMapper
  -> parcel tokens [B, P, D]
  -> zero / training-mean / no intervention
  -> optional TokenMapper
  -> diffusion condition tokens
```

parcel 索引不得作用于 decoder-query tokens。训练集均值替换使用对应
parcel 的 ParcelMapper 输出在训练集上的均值。

## 对比条件

每项确认性 ROI 分析均包括：

- 不屏蔽基线（no-mask）；
- 完整 ROI 组的零值替换和均值替换；
- 等数量（equal-k）的零值替换和均值替换；
- 按半球、SNR 和 parcel 大小匹配的随机对照；
- 至少一个无关功能 ROI 对照。

Face、Body 和 Scene 属于确认性分析。Word 保持探索性分析，除非至少有
20 个经过审查的刺激可用。

当前 E2 pilot 先只执行 zero-mask。训练集均值替换保留为后续稳健性分析，
不属于本轮 30 图 pilot 的运行条件。

## 来源与可追溯性

每次运行都必须记录：仓库 commit、checkpoint 及其哈希、映射及其哈希、
所选 parcel 索引、subject、数据集索引、随机种子、扩散参数、干预阶段
与模式、被屏蔽索引，以及软件环境。

大尺寸图片、checkpoint、数据集、缓存和完整日志不进入 Git。小型
CSV/JSON 摘要和审查图可以提交。
