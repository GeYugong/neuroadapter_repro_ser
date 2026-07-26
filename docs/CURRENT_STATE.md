# 当前研究状态

最后更新：2026-07-26

## 当前范围

阶段 A、B、30 图 E2 pilot、正式 E2 zero-mask、预注册 E2b mean-mask、
E3a 类别×ROI 交互和 E3b 联合 ROI 冗余的三 seed 正式实验均已完成。
没有为 E3 训练新模型，也没有根据正式结果追加新条件。
附录 P 复现尝试以及已有的 50 样本 zero-mask 输出保留在
`experiments/roi_ablation/` 下，作为历史性/探索性工作。

正式 E2b 推理使用的代码提交为
`da3de7c853f9504cb0dd3eefebcabb2eda6515aa`；包含最终结果、图表和解释
复核的报告快照为 `f7b9603bdf73c8b53c59cf86ad4c444b2220594c`。
前者用于追溯实际运行代码，后者用于追溯阶段关闭时的报告状态。

## 阶段 A：实现

- 已建立研究计划、实验协议、决策日志、结果模板、Python 包结构、路径
  示例和 E2 pilot 设计。
- ROI 干预现在作用于 ParcelMapper 输出，并位于可选 TokenMapper 之前。
- `none`、`zero` 和训练集均值替换会检查 parcel 维度与索引，保证非目标
  parcel 在比特级完全不变，并记录 token norm 审计结果。
- 均值替换使用训练集上的 ParcelMapper 输出，而不是 decoder query。
- 未修改上游 NeuroAdapter checkout。
- 已在项目目录建立独立 E3 测试环境，同时提供 pytest、scikit-image
  和 OpenCV Haar API。正式运行前完整测试结果为 `52 passed`，没有 ignore 旧 E2
  指标测试，也没有修改共享 conda 环境。
- 真实 step-100000 checkpoint 的 smoke test 已通过。其 `sub_approach`
  为 `linear_projection`：fMRI `[1, 200, 626]` 被映射为
  `[1, 200, 768]` 的 parcel token 和 condition token，不经过
  TokenMapper。no-mask 与上游 forward 路径完全一致，zero masking
  只改变指定 parcel。
- 可选 Transformer decoder 路径另有单元测试覆盖；当前 checkpoint
  并不使用该架构。

## 阶段 B：E0 映射

主要映射采用公开的 Algonauts Project 2023 Subject 1 fsaverage 数据，
并使用严格大于 0.5 的重叠规则。

| ROI | 全部 1000 parcels | Top-SNR-200 | 保留率 |
| --- | ---: | ---: | ---: |
| V1 | 10 | 10 | 100% |
| V2 | 9 | 9 | 100% |
| V3 | 6 | 6 | 100% |
| V4 | 5 | 5 | 100% |
| Face | 4 | 4 | 100% |
| Body | 24 | 24 | 100% |
| Scene | 31 | 31 | 100% |
| Word | 8 | 8 | 100% |
| 未标注 | 903 | 103 | 11.4% |

通过公开数据获得功能 ROI 标签的 97 个 parcel 全部已进入 top-SNR-200。
在这一映射下，top-SNR 选择**没有**降低 Face、Word 或 V4 的覆盖率。
相对于全部 parcel 中 20% 的总体选择比例，它反而显著富集了有功能标签
的 parcel。因此，原先基于覆盖不足提出的 ROI-balanced-200 模型训练
前提不成立。

生成产物：

```text
experiments/E0_mapping/
  algonauts_full_parcel_inventory_subj01.csv
  algonauts_top200_mapping_subj01.csv
  roi_coverage_all_vs_top200.csv
  mapping_metadata.json
  file_hashes.json
  figures/
```

## 阶段 B：E1 刺激筛选

筛选过程只使用 NSD ground-truth 刺激。CLIP RN50 提供语义分数，OpenCV
Haar 提供人脸几何证据，COCO 2017 官方实例标注提供 person segmentation
area。清单记录了依赖来源、哈希、软件版本、阈值和推理参数。

| 类别 | 候选数 | 选中数 | 用途 |
| --- | ---: | ---: | --- |
| Face | 63 | 37 | 确认性；低于期望下限 40 |
| Body | 136 | 50 | 确认性 |
| Scene | 387 | 50 | 确认性 |
| Word | 24 | 21 | 仅探索性；缺少 OCR 证据 |

审查图不是只选择最高分样本，而是在完整分数排序中等间距取样。视觉检查
确认 Face、Body 和 Scene 的审查样本可用。Word 中存在 CLIP 误检，因此
不能进入确认性检验。其 21 个入选样本不包含任何确认性清单中的数据集
索引。为达到目标数量，没有复制任何样本。

生成产物：

```text
experiments/E1_stimulus_manifest/
  all_test_images.csv
  face_candidates.csv
  body_candidates.csv
  scene_candidates.csv
  word_candidates.csv
  confirmatory_manifest.csv
  exploratory_manifest.csv
  figures/category_audit_grid.png
  manifest_metadata.json
```

## E2 zero-mask pilot

已完成 Face、Body、Scene 各 10 张图、seed 12345、50 步扩散的 zero-mask
pilot。新运行器可读取非连续 manifest 索引；随机对照按数量、半球、SNR
和 parcel 大小匹配；初始及逐步 DDPM 噪声均在条件间共享。

- 完整测试：`23 passed`；
- 30/30 个 no-mask 跨 batch 重复图像 SHA-256 一致；
- 非目标 parcel 最大变化量：`0.0`；
- Face PixCorr 正向信号：excess `0.00696`，`q=0.091`；
- Scene full DINO 正向信号：excess `0.04394`，`q=0.060`；
- Body 没有跨指标一致证据；
- 没有结果达到 `q<0.05`。

详细指标和视觉对比见 `experiments/E2_zero_pilot/`。pilot 证明实验管线
可用，但样本量不足以形成正式功能特异性结论。

## 正式 E2 zero-mask 结果

正式实验使用 Face 37、Body 50、Scene 50 和 3 个预注册 seed，共完成
411 个 image-seed pairs 和 5499 个条件干预。

- 9/9 个运行完整；
- 411/411 个确定性 SHA-256 检查通过；
- 5499/5499 个 intervention audit 完整；
- 非目标 parcel 最大变化量：`0.0`；
- 主要统计先按图像平均 3 seeds，再对 15 项检验统一 BH 校正。

没有校正显著的正向结果支持类别匹配 ROI 比匹配随机 parcel 造成更大
重建下降。唯一 `q<0.05` 的 Body SSIM excess 为负，方向与假设相反。
pilot 中 Face PixCorr 和 Scene DINO 的信号未在正式实验中复现。

完整结果与视觉审查图见 `experiments/E2_zero_full/`。

## E2b mean-mask 稳健性分析

E2b 在查看 mean 结果前预注册，只把 parcel 干预值从全零改为 Subject 1
的 9000 个训练样本在 `ParcelMapper` 输出上的逐 parcel 均值。

- zero recheck 主要 excess 最大变化 `4.22e-6`，没有 q 值跨越 0.05；
- mean cache 为 `[200,768]`，无 NaN/Inf，checkpoint 与 selected
  parcel 哈希验证通过；
- zero/mean plan 的 dataset、target、random、unrelated indices 完全相同；
- smoke 3/3 和正式 9/9 runs 通过；
- 正式完成 411/411 确定性检查和 5499/5499 干预审计；
- 非目标 parcel 最大变化为 `0.0`；
- 每个 mean no-mask 与对应 zero no-mask 的 PNG SHA 完全一致。

mean 的 15 项主要检验没有校正显著的正向结果。Face PixCorr 和 Body
CLIP 的未校正 p 分别约为 0.042 和 0.017，但全局 q 为 0.315 和 0.258。
Body equal-k CLIP 为校正显著的负向 excess，不支持原假设。

zero 与 mean 有 10/15 项同方向，其中 4 项共同为正、6 项共同为负；
整体 Pearson 为 `0.230`，Spearman 为 `0.400`。15 项 mean-zero 配对差异
均未通过次要 BH 校正。上述相关性仅为描述性结果：总体只有 15 个点，
每个类别内部只有 5 个点，每个指标内部只有 3 个点，不能把类别内或
指标内的高相关系数作为强统计证据。

按预定义规则属于情况 A：在当前公开 ROI 映射、当前 checkpoint 和两种
parcel 干预方式下，没有获得稳健的类别匹配 ROI 额外因果贡献证据。
这不等于这些脑区没有功能。

## 阶段关闭与下一研究决策

zero-mask 阶段曾存在“全零 token 属于分布外干预”的疑问；该问题现已
通过 E2b mean-mask 稳健性实验进行验证。当前能够形成的模型层面结论是：
这个已训练的 NeuroAdapter 模型没有表现出能被单组 ROI 整体消融稳定
检测到的类别匹配依赖。

当前仍有四项关键限制：

1. 59 个目标 parcel 中有 8 个同时对目标组和非目标组达到 0.5 overlap，
   因此目标 ROI 并不纯净；
2. frozen random controls 中存在目标 ROI overlap，且审计计数是不同
   replicate 中的 control 记录数，不等于独立 parcel 数；
3. 整图指标可能稀释面部、人体或背景区域的局部变化；
4. 单 ROI 干预不能直接检验多个脑区之间的信息冗余或生成先验的补偿。

## E3 三 seed 正式实验

E3 的工程 smoke、严格 Haar 修复和每类 10 张单 seed pilot 完成后，在
查看正式结果前冻结了全量 plan。实际推理与评价代码提交为
`de5a68f61ee31f37cb6ed0b8eacf607c52869183`。正式配置没有根据 pilot
结果调整：

- Face 37、Body 50、Scene 50；
- seeds `12345`、`23456`、`34567`；
- training-mean replacement、equal-k=4；
- 每个目标条件 5 组唯一 pure matched-random controls；
- pure control 对所有目标 ROI 的 overlap 严格 `<0.10`；
- 同一 checkpoint、mean cache、Haar detector、50 步扩散和评价指标。

完整回归测试为 `52 passed`。E3a/E3b 的 plan equivalence 和 control
matching audits 均通过；正式 plan 的 checkpoint、mean cache 和 Haar
SHA 分别为：

```text
checkpoint:
2d340552270db08a8518fd60949af1fa1b823ac4fd1d18eab7b17a0d04ec3a40
mean cache:
283159cd0f610202b7ebfb60e85a97ad3a49af6662bcbb364239375b9b228d1e
Haar cascade:
0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0
```

| 实验 | 任务 | image-seed pairs | 条件/图 | 记录 | 评价 |
| --- | ---: | ---: | ---: | ---: | --- |
| E3a interaction | 9/9 | 411 | 20 | 8220 | PASS |
| E3b joint redundancy | 9/9 | 411 | 32 | 13152 | PASS |

两项实验的 no-mask SHA、共享 initial latent/noise、checkpoint、mean
cache、GT、图片、seed 和 condition 与冻结 plan 完全一致；非目标 parcel
最大变化均为 `0.0`。评价不存在缺图、错位、NaN、非法指标或空局部区域，
最小局部区域为 3025 pixels。Face 预测检测成功数分别为 1093/2220 和
1772/3552。

### E3a 结果

主要统计先在每张图片内平均 3 个 seed，再计算某 ROI 在匹配类别上的
causal loss 减两个非匹配类别的等权平均 causal loss。15 项全局检验统一
BH，27 项局部检验作为独立统计族 BH。

全局和局部统计族均没有 `q<0.05` 的结果。最小的全局 q 来自 Body CLIP，
效应为 `-0.01347`，p=`0.00920`，q=`0.13799`，方向与类别匹配假设相反。
Face PixCorr 为正 `0.00436`，但 q=`0.35809`。正式结果不支持稳健的
类别×ROI 交互。

### E3b 结果

E3b 分别比较匹配单 ROI、三个双 ROI 和三 ROI 联合干预与相同 parcel
数量的 pure controls。全局 75 项、局部 45 项、全局趋势 15 项和局部
趋势 9 项分别独立 BH。

120 项联合检验有 5 项 `q<0.05`，但方向混合：3 项全局结果中 2 项为负，
唯一正向 Body Face+Scene CLIP 条件不包含匹配 Body ROI；2 项局部正向
结果为 Face 三 ROI face LPIPS 和 Scene 三 ROI scene consistency。
它们是候选联合效应，但没有一致的全局支持。

24 项规模趋势只有 Face SSIM 通过校正，4/8/12 parcels 的 excess 依次为
`0.00087`、`-0.00160`、`-0.00515`，斜率 `-0.00301`，q=`0.04200`。
方向与预注册的“联合规模增加时损失增强”假设相反。局部趋势无显著结果。

### 人工审图与阶段结论

E3a/E3b 共 6 张 comparison grid、274 行样本已经逐行检查。所有图片非空，
GT、dataset index 和条件列对齐，无损坏、错列或异常 fallback。可见变化
具有明显图像间异质性，没有一致的类别匹配模式或 4/8/12 parcels 单调
增强模式。

正式结果位于：

```text
experiments/E3_interaction_full/
experiments/E3_joint_redundancy_full/
```

当前结论是：在现有 NeuroAdapter checkpoint 和公开 ROI 映射下，没有
获得稳健的类别特异交互或联合规模单调效应证据。两个三 ROI 局部结果只
能作为后续独立验证候选。按预注册边界停止，不训练新模型、不追加条件，
也不根据正式结果重新定义统计。
