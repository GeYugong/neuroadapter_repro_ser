# 决策记录

## D001：公开 ROI 实验协议

主要 ROI 映射采用 Algonauts Project 2023 发布的 Subject 1 fsaverage
数据，并使用严格大于 0.5 的重叠阈值。由此得到的 top-SNR-200 映射被
视为一套基于公开数据的研究协议，而不是对附录 P 的严格复现。

## D002：历史实验

已有 ROI 复现产物和 50 样本 zero-mask pilot 均予以保留，但其结论标记
为历史性/探索性结果。

## D003：干预阶段

ROI 干预作用于 ParcelMapper 的输出，并位于 TokenMapper 之前。原因是
transformer decoder 会把 200 个 parcel token 映射为 50 个 decoder
query，使二者不再具有一一对应的索引关系。

实际的 step-100000 checkpoint 使用 `linear_projection`，而不是
`transformer_decoder`；它的 200 个 ParcelMapper 输出被直接用作
200 个 condition token。为兼容 Transformer checkpoint，TokenMapper
之前的干预实现仍然有必要，并且该分支已有单元测试覆盖。

## D004：刺激分类

刺激清单生成器不会自行下载权重。语义置信度使用调用方提供的 OpenAI
CLIP RN50 checkpoint。Face 的几何证据使用显式提供的 OpenCV 4.12
Haar cascade，并要求 COCO 中存在 person。Body 和 Scene 使用 COCO 2017
官方实例标注中的 person segmentation area，而不使用出现明显误检的
OpenCV HOG。由于缺少 OCR 证据，Word 保持探索性分析。

## D005：top-SNR 覆盖假设

E0 完整清单显示，通过公开数据标注的 97 个功能 parcel 全部已进入
top-SNR-200。Face、Word 和 V4 的保留率均为 100%。因此，在主要公开
映射下，H3 中“部分 ROI 覆盖不足”的版本不成立。仅凭覆盖率无法证明
有必要训练 ROI-balanced 模型；除非预先注册新的选择偏差问题，否则
暂停该训练。

## D006：如实记录刺激数量

最终确认性清单包含 37 张 Face、50 张 Body 和 50 张 Scene 图像。Face
采用保守规则，宁可比期望下限 40 少 3 张，也不保留审查出的误检或重复
图像。24 个仅由 CLIP 识别的 Word 候选中，有 21 个不与确认性清单重叠，
因此保留为探索性样本。由于缺少必要的 OCR 证据，它们不进入确认性分析。

## D007：阶段 C 准入条件

旧版解码器无法读取刺激清单中不连续的数据集索引，旧版随机对照生成器
也无法读取 E0 的数据结构。因此二者当前都不能直接作为 E2 runner。
阶段 C 必须先实现基于清单的数据选择，以及经过验证的 zero/mean 匹配
随机对照。

## D008：E2 pilot 以 zero-mask 为主

当前 E2 pilot 只运行 zero-mask，不生成 mean-mask 条件，也不计算训练集
parcel mean。mean replacement 的代码能力予以保留，待 zero-mask pilot
完成并证明实验流程与效应方向值得继续后，再作为稳健性分析单独启用。

## D009：正式 E2 统计方案

正式 E2 使用 Face 37、Body 50、Scene 50 张图和 3 个固定 seed：
`12345`、`23456`、`34567`。主要分析使用各类别完整 ROI，并与 5 组相同
parcel 数量、半球、SNR 和大小匹配的随机对照比较。

统计单位是图像，不把 3 个 seed 当作独立样本。先对每张图的 3 个 seed
causal loss 求平均，再进行 bootstrap CI 和 sign-flip 检验。主要检验包含
3 类 × 5 指标共 15 项，统一进行 Benjamini-Hochberg 校正。Body/Scene 的
equal-k=4 结果只作为次要敏感性分析，不能替代主要 full-group 结果。

## D010：E2b mean-mask 稳健性实验

正式 zero-mask 的 15 项主要检验没有提供校正显著的正向证据。由于全零
parcel token 可能偏离训练分布，E2b 使用训练集 parcel-wise mean 进行
替换，以判断正式负结果是否依赖干预定义。

mean token 定义为 Subject 1 训练集中每个样本经过 `ParcelMapper` 后所得
`[200, 768]` token 在样本维度上的均值。干预位置固定为
`after_parcel_mapper_before_token_mapper`，不得使用测试集、验证集、
condition token、decoder query 或重建输出计算均值。

E2b 与正式 zero-mask 使用相同 checkpoint、测试图、dataset indices、
目标 ROI、5 组 matched-random controls、unrelated ROI、3 个 seed、
扩散参数、latent/noise 配对、五项指标及统计单位。主要检验仍是
3 类别 × 5 指标，共 15 项全局 BH 校正；不得根据结果调整范围或改用
单侧检验。

zero 与 mean 均无校正显著正向结果时，只能认为当前映射、checkpoint 和
两种干预均未提供稳健证据。仅 mean 出现正向结果时，应解释为对干预方式
敏感的候选效应；二者方向相反时不得形成稳健功能解释。正式 zero 结果
不可覆盖、删除或重新选择性分析。
