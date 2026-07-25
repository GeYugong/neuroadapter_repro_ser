# 正式 zero-mask 统一评价器复核

本目录使用新的内容寻址评价器重新计算既有正式 zero 输出，不覆盖
`experiments/E2_zero_full/`。

- no-mask 与 no-mask-repeat 的 PNG SHA 以及五项指标逐值一致；
- 样本数、seed 数与 condition 集合一致；
- 主要 target-random excess 最大绝对变化约为 `4.22e-6`；
- 没有任何 q 值跨越 0.05；
- PixCorr/SSIM 完全一致；
- LPIPS、CLIP、DINO 的极小差异来自统一的哈希缓存与 batch 计算顺序。

`zero_recheck_comparison.json` 给出旧版与新版的逐类别、逐指标比较。该
复核通过了 E2b 正式运行前规定的停止门禁。
