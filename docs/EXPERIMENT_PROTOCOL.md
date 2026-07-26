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

30 图 pilot 和正式 E2 已按 zero-mask 完成。E2b 是独立预注册的
mean-mask 稳健性分析；它必须复用正式 zero plan 中的样本和全部 parcel
indices，只允许替换干预模式与条件名后缀。正式 zero 结果不可覆盖。

E2b 的主要统计单位仍为图像。同一图像先平均 3 个 seed 的 causal loss，
再计算双侧 sign-flip p、bootstrap 95% CI，并对 3 类别 × 5 指标共
15 项主要检验统一进行 Benjamini-Hochberg 校正。

E3a 使用 3×3 刺激类别×被干预 ROI 设计，所有 ROI 固定 equal-k=4，
并使用 overlap `<0.10` 的 pure controls。E3b 使用匹配单 ROI、三个双
ROI、三 ROI 联合干预及其相同 parcel 数量 controls。两项实验均先在
图像内平均 3 个 seed；全局和局部指标分别构成独立 BH 统计族。E3b 趋势
固定为 4、8、12 parcels 三个 level，不得根据正式结果修改。

## 来源与可追溯性

每次运行都必须记录：仓库 commit、checkpoint 及其哈希、映射及其哈希、
所选 parcel 索引、subject、数据集索引、随机种子、扩散参数、干预阶段
与模式、被屏蔽索引，以及软件环境。

大尺寸图片、checkpoint、数据集、缓存和完整日志不进入 Git。小型
CSV/JSON 摘要和审查图可以提交。
