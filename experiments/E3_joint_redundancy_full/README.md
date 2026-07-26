# E3b 联合 ROI 冗余正式实验

## 研究问题

E3b 检验单个 ROI 的作用是否因多个脑区之间的信息冗余而难以检测。实验
使用与 E3a 相同的 Face 37 张、Body 50 张、Scene 50 张冻结图片和 3 个
预注册 seed。每类图片包含：

- 类别匹配的单 ROI 干预，4 parcels；
- Face+Body、Face+Scene、Body+Scene，8 parcels；
- Face+Body+Scene，12 parcels；
- 每个目标条件对应 5 组相同 parcel 数量的 pure matched-random controls。

干预使用训练集 parcel mean replacement，pure controls 对所有目标 ROI
的 overlap 严格 `<0.10`。主要比较是联合 ROI causal loss 减去相同
parcel 数量 pure controls 的平均 causal loss。

趋势检验在查看正式结果前固定为：

```text
level 1: 匹配单 ROI，4 parcels
level 2: 两个包含匹配 ROI 的双 ROI 条件之平均，8 parcels
level 3: Face+Body+Scene，12 parcels
```

每张图片先平均 3 个 seed，再计算三个 level 上的最小二乘斜率。全局联合
干预 75 项、局部联合干预 45 项、全局趋势 15 项和局部趋势 9 项分别构成
四个独立 BH 统计族。

## 完成与审计

| 项目 | 结果 |
| --- | ---: |
| category-seed 任务 | 9/9 |
| image-seed pairs | 411 |
| 条件/图 | 32 |
| condition-image records | 13152 |
| per-sample 全局指标 | 13152 |
| per-sample 局部指标 | 13152 |
| 联合干预正式检验 | 120 |
| 单调趋势正式检验 | 24 |
| 缺图、非法指标、NaN、空局部区域 | 0 |
| 非目标 parcel 最大变化 | 0.0 |

所有 plan、pure controls、图片、GT、condition、seed、checkpoint、
mean cache、共享 latent/noise 和确定性审计通过。Face 预测图的人脸检测
成功数为 1772/3552；冻结局部区域均非空，最小区域为 3025 pixels。

## 联合干预结果

120 项联合干预检验中有 5 项 `q<0.05`：

| 统计族 | 图片 | 干预 | 指标 | 效应 | 95% bootstrap CI | p | BH q |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| 全局 | Body | Face+Body | CLIP | -0.01326 | [-0.02068, -0.00662] | 0.00030 | 0.01750 |
| 全局 | Body | Face+Scene | CLIP | 0.01351 | [0.00611, 0.02218] | 0.00070 | 0.01750 |
| 全局 | Scene | Body+Scene | SSIM | -0.00540 | [-0.00856, -0.00245] | 0.00070 | 0.01750 |
| 局部 | Face | 三 ROI | face LPIPS | 0.01055 | [0.00468, 0.01682] | 0.00150 | 0.04275 |
| 局部 | Scene | 三 ROI | scene consistency | 0.05333 | [0.02267, 0.08533] | 0.00190 | 0.04275 |

结果方向混合。Body 的 Face+Body CLIP 和 Scene 的 Body+Scene SSIM 为负，
不支持匹配 ROI 联合干预造成更大损失。Body 图片的 Face+Scene CLIP 为
正，但该条件不包含匹配的 Body ROI，不能作为类别匹配冗余证据。两个三
ROI 局部结果为正，是值得保留的候选联合效应，但没有获得对应的全局指标
一致支持。

## 单调趋势结果

24 项趋势检验只有 Face SSIM 通过校正：

| 图片 | 指标 | k=4 | k=8 | k=12 | 每 level 斜率 | 95% bootstrap CI | p | BH q |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Face | SSIM | 0.00087 | -0.00160 | -0.00515 | -0.00301 | [-0.00509, -0.00112] | 0.00280 | 0.04200 |

该斜率为负，方向与“联合 ROI 数量增加时 causal loss 增强”的预注册假设
相反；其逐图非递减比例仅为 `0.0811`。局部趋势没有任何校正显著结果。
因此，正式实验不支持联合规模增加导致效应单调增强。

## 人工审图

3 张 comparison grid 共 137 行，已逐行检查。所有图片非空，GT、
dataset index 和条件列对齐，没有损坏、错列或 fallback。联合干预在部分
样本上造成较明显的人物、物体或背景变化，但不同样本和不同条件的方向
不一致，没有形成普遍的、随 4/8/12 parcels 单调增强的视觉模式。

![Face comparison](figures/face_comparison_grid.png)

![Body comparison](figures/body_comparison_grid.png)

![Scene comparison](figures/scene_comparison_grid.png)

![Joint-size trend](figures/joint_size_trend_plot.png)

![Global joint effects](figures/global_effect_forest_plot.png)

![Local joint effects](figures/local_effect_forest_plot.png)

![Selected effects](figures/effect_forest_plot.png)

![Effect distributions](figures/effect_distribution_plot.png)

## 结论

E3b 发现两个三 ROI 局部候选效应，但整体结果方向混合，且唯一校正显著
的规模趋势方向与假设相反。当前证据不足以支持“NeuroAdapter 依赖多个
高层功能 ROI 的分布式冗余信息，且联合消融越大重建损失越大”。候选局部
效应应作为后续独立验证假设，不能据此追加本轮条件或重新定义趋势。

## 关键产物

- `plan.json`：正式单/联合 ROI、pure controls、seed 和资产哈希；
- `plan_equivalence_audit.json`、`control_matching_audit.json`；
- `output_audit.json`、`evaluation_audit.json`、`visual_review.json`；
- `per_sample_metrics.csv`、`local_metric_results.csv`；
- `per_image_excess_effects.csv`；
- `joint_mask_results.csv`：120 项联合干预正式检验；
- `monotonicity_results.csv`：24 项预注册趋势检验。

实际推理与评价代码提交为
`de5a68f61ee31f37cb6ed0b8eacf607c52869183`。最终报告提交另行记录，
不改变本次冻结分析。
