# E2b mean-mask 正式稳健性实验

本实验与正式 zero-mask 使用相同 Subject 1、step-100000 checkpoint、
Face 37、Body 50、Scene 50、3 个 seed、冻结 ROI/random controls、
50 步扩散与 factor 4.0。唯一主要变化是以 9000 个训练样本的
parcel-wise mean 替换指定 token。

## 工程验收

- plan 等价审计通过；
- smoke 3/3 图片、39/39 条件通过；
- 正式 9/9 runs、411/411 确定性检查、5499/5499 干预审计通过；
- 非目标最大变化为 `0.0`；
- 所有运行使用同一 checkpoint、mean cache 和代码提交；
- zero/mean 的 no-mask PNG SHA 完全一致；
- 三张完整 comparison grid 逐行检查无空白、损坏、错位或条件列异常。

版本区分：

```text
inference code commit: da3de7c853f9504cb0dd3eefebcabb2eda6515aa
reviewed reporting snapshot: f7b9603bdf73c8b53c59cf86ad4c444b2220594c
```

前者是 9 个正式任务实际使用的代码版本，已经由运行审计逐任务确认；
后者包含正式指标、图表和报告。后续提交只修正文档解释，不改变推理输出
或统计数字。

## 主要结果

15 项主要检验经全局 BH 校正后，没有校正显著的正向结果。最小的正向
未校正 p 为 Body CLIP（excess `0.01667`，p `0.0172`，q `0.2580`）；
Face PixCorr 为 excess `0.00290`、p `0.0420`、q `0.3154`。二者均不
达到预注册的校正阈值。

表中 95% bootstrap CI 和单项 p 值均未对 15 项多重比较进行调整，正式
判断以预注册的全局 Benjamini-Hochberg q 值为准。因此 Face PixCorr
和 Body CLIP 即使普通 95% CI 不包含 0，也不能作为正式阳性结果。

Body equal-k CLIP 为显著负向 excess（约 `-0.01104`），方向不支持
类别匹配 ROI 假设。其他 equal-k 指标也没有校正显著的正向结果。

## Zero 与 mean

15 项中 10 项方向一致，4 项共同为正，6 项共同为负。整体 Pearson
相关为 `0.230`，Spearman 为 `0.400`；15 项 mean-zero 配对差异均未
通过独立的次要 BH 校正。这些相关性只作描述性汇总：总体只有 15 个点，
每个类别只有 5 个点，每个指标只有 3 个点，类别内或指标内的高相关系数
不能解释为强统计证据。

结果属于预定义情况 A：在当前公开 ROI 映射、当前 checkpoint 和
zero/mean 两种 parcel 干预下，没有获得稳健的类别匹配 ROI 额外因果
贡献证据。这不等于对应脑区没有生物学功能。

更准确的模型层面表述是：当前这个已经训练好的 NeuroAdapter 模型，
没有表现出能够被单组 ROI 整体消融稳定检测到的类别匹配依赖。该结果
不能证明 fMRI 中不存在类别信息，也不能外推到其他模型或所有被试。

## 关键限制

- 59 个目标 parcel 中有 8 个同时与目标和非目标功能组高度重叠，单组
  ROI 干预并不等于只删除一种类别信息；
- 例如 Face token 16 同时高度覆盖 Face/Body，token 22 覆盖 Face/V3，
  token 152 覆盖 Face/Word，只有 token 43 相对更纯；
- frozen controls 的目标 overlap 计数是 control 记录数，同一 parcel
  可在不同 replicate 中重复出现，不能当作独立 parcel 数量；
- 整图指标可能稀释人脸、人体或背景区域中的局部变化；
- 单 ROI 干预不能直接回答多个脑区之间是否存在分布式冗余。

可行性审计表明，在目标 overlap `<0.10` 时三类仍可构造 5 组唯一
pure controls，但本阶段没有事后替换正式 controls。

详细数字见 `primary_results.csv`、`secondary_equal_k_results.csv`、
`zero_vs_mean_robustness.csv` 和 `zero_vs_mean_summary.json`。

## 图表

![Mean-mask 主要效应与置信区间](figures/primary_effect_forest_plot.png)

![Zero 与 mean 效应比较](figures/zero_vs_mean_effect_scatter.png)

![目标 ROI overlap](figures/target_roi_purity_plot.png)

![随机对照污染分布](figures/control_contamination_plot.png)

![Face 正式重建对比](figures/face_comparison_grid.jpg)

![Body 正式重建对比](figures/body_comparison_grid.jpg)

![Scene 正式重建对比](figures/scene_comparison_grid.jpg)
