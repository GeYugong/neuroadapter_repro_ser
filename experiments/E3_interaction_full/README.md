# E3a 类别×ROI 交互正式实验

## 研究问题

E3a 检验同一功能 ROI 的干预效应是否在其匹配刺激类别上更强。实验使用
Face 37 张、Body 50 张、Scene 50 张冻结图片，以及预注册的 3 个 seed：
`12345`、`23456`、`34567`。

每张图片分别干预 Face、Body、Scene ROI。所有 ROI 均固定为
`equal-k=4`，使用 Subject 1 训练集 parcel mean replacement；每个目标
条件配 5 组目标 ROI overlap 严格 `<0.10` 的 pure matched-random
controls。单张图片的所有条件共享 initial latent 和 diffusion noise。

主要交互统计先在每张图片内平均 3 个 seed，再计算：

```text
某 ROI 在匹配类别上的 target-minus-random causal loss
-
该 ROI 在另外两个非匹配类别上的等权平均 causal loss
```

两个非匹配类别先分别求均值，再等权平均，因此 Body 和 Scene 较大的样本
数不会压过 Face。5 个全局指标的 15 项检验统一进行 BH 校正；27 项局部
类别内 target-vs-random 检验构成独立 BH 统计族。

## 完成与审计

| 项目 | 结果 |
| --- | ---: |
| category-seed 任务 | 9/9 |
| image-seed pairs | 411 |
| 条件/图 | 20 |
| condition-image records | 8220 |
| per-sample 全局指标 | 8220 |
| per-sample 局部指标 | 8220 |
| 正式全局检验 | 15 |
| 正式局部检验 | 27 |
| 缺图、非法指标、NaN、空局部区域 | 0 |
| 非目标 parcel 最大变化 | 0.0 |

plan equivalence、pure-control matching、输出完整性和评价审计全部通过。
411/411 个 no-mask 确定性检查通过；同图全部条件共享的 latent/noise、
checkpoint、mean cache、Haar detector、图片、GT、seed 和 condition
均与冻结 plan 一致。Face 预测图的人脸检测成功数为 1093/2220；检测失败
时使用从 GT 冻结的非空局部区域，最小区域为 3025 pixels。

## 正式结果

15 项全局交互检验和 27 项局部检验均没有 `q<0.05` 的结果。

最接近的全局结果为：

| ROI | 指标 | 交互效应 | 95% bootstrap CI | p | BH q |
| --- | --- | ---: | --- | ---: | ---: |
| Body | CLIP | -0.01347 | [-0.02232, -0.00459] | 0.00920 | 0.13799 |
| Body | DINO | -0.01255 | [-0.02281, -0.00260] | 0.03150 | 0.23623 |
| Face | PixCorr | 0.00436 | [0.00049, 0.00841] | 0.09549 | 0.35809 |

Body 的两个候选效应为负，方向与“匹配类别受影响更强”的预期相反。
Face PixCorr 为正，但未通过全局多重比较校正。

局部指标中，Face 图片 mask Face 的 face LPIPS 为 `0.00424`
（p=`0.03600`，q=`0.32397`）；其余局部结果同样没有通过独立 BH 校正。
因此，正式实验没有提供稳健的类别×ROI 交互证据。

## 人工审图

3 张 comparison grid 共 137 行，已逐行检查。GT、dataset index 和条件列
对齐，图片均非空，没有损坏、错列、标题遮挡或异常 fallback。干预造成的
视觉变化在样本间差异较大：部分图片的人物、物体或背景结构明显变化，
另一些几乎不变，没有形成跨样本一致的类别匹配模式。

![Face comparison](figures/face_comparison_grid.png)

![Body comparison](figures/body_comparison_grid.png)

![Scene comparison](figures/scene_comparison_grid.png)

![Global interaction effects](figures/effect_forest_plot.png)

![Local effects](figures/local_effect_forest_plot.png)

![Effect distributions](figures/effect_distribution_plot.png)

## 结论

在当前公开 ROI 映射、step-100000 checkpoint、训练集 mean replacement、
equal-k pure controls 和三 seed 正式设计下，没有观察到可通过多重比较
校正的类别特异 ROI 交互。该结论只针对当前 NeuroAdapter 模型的可检测
依赖，不代表这些功能脑区没有生物学功能。

## 关键产物

- `plan.json`：正式刺激、parcel、条件、seed、模型资产及哈希；
- `plan_equivalence_audit.json`：与 E2 冻结设置的等价性；
- `control_matching_audit.json`：pure controls 的纯度与匹配审计；
- `output_audit.json`：运行、确定性、共享随机状态和干预审计；
- `evaluation_audit.json`：指标、局部区域和正式推断完整性；
- `visual_review.json`：全部 comparison grid 的人工检查记录；
- `per_sample_metrics.csv`、`local_metric_results.csv`：逐条件指标；
- `per_image_excess_effects.csv`：按图像聚合的 excess causal loss；
- `interaction_results.csv`：15 项全局正式检验；
- `local_interaction_results.csv`：27 项局部正式检验。

实际推理与评价代码提交为
`de5a68f61ee31f37cb6ed0b8eacf607c52869183`。最终报告提交另行记录，
仅包含结果、图片和文档，不改变本次冻结分析。
