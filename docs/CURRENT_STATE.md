# 当前研究状态

最后更新：2026-07-25

## 当前范围

阶段 A、B、30 图 E2 pilot、正式 E2 zero-mask 和预注册 E2b mean-mask
稳健性实验均已完成。没有训练新模型。
附录 P 复现尝试以及已有的 50 样本 zero-mask 输出保留在
`experiments/roi_ablation/` 下，作为历史性/探索性工作。

## 阶段 A：实现

- 已建立研究计划、实验协议、决策日志、结果模板、Python 包结构、路径
  示例和 E2 pilot 设计。
- ROI 干预现在作用于 ParcelMapper 输出，并位于可选 TokenMapper 之前。
- `none`、`zero` 和训练集均值替换会检查 parcel 维度与索引，保证非目标
  parcel 在比特级完全不变，并记录 token norm 审计结果。
- 均值替换使用训练集上的 ParcelMapper 输出，而不是 decoder query。
- 未修改上游 NeuroAdapter checkout。
- 服务器验证结果：`23 passed`。
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

## 当前限制与后续问题

1. 结论只适用于公开 Algonauts ROI 映射和当前 step-100000 checkpoint；
2. 作者附录 P 使用的原始 ROI metadata 仍未公开；
3. mean-mask 尚未运行，zero-mask 可能产生分布外 token；
4. 在没有新的预注册假设前，不应继续根据现有结果反复调整 ROI 或指标。

## 下一步准确命令

以下命令可安全地重新验证已经完成的 A/B 阶段准入条件：

```bash
REPRO_ROOT="$(git rev-parse --show-toplevel)"
PROJECT_ROOT="$(dirname "$REPRO_ROOT")"
cd "$REPRO_ROOT"
PYTHONPATH="$PROJECT_ROOT/tools/test-deps:src" \
  conda run -n neuroadapter python -m pytest -q
python scripts/validate_stage_ab_artifacts.py
conda run -n neuroadapter python scripts/smoke_test_checkpoint_intervention.py \
  --checkpoint "$PROJECT_ROOT/outputs/neuroadapter/20260707-topk100-bs4-ddp4-resume50000-to100000/checkpoint-step-100000.pt" \
  --upstream-root "$PROJECT_ROOT/code/NeuroAdapter"
```

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
均未通过次要 BH 校正。

按预定义规则属于情况 A：在当前公开 ROI 映射、当前 checkpoint 和两种
parcel 干预方式下，没有获得稳健的类别匹配 ROI 额外因果贡献证据。
这不等于这些脑区没有功能。
