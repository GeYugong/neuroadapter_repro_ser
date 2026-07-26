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

## D011：E3 先完成工程 smoke，不自动进入 pilot

E3a 使用完整 `3 × 3` 刺激类别×被干预 ROI 设计，三个 ROI 均固定为
equal-k=4。E3b 分别检验类别匹配单 ROI、三个双 ROI 组合和三 ROI 联合
干预。干预值继续使用训练集 parcel mean。

所有随机对照必须满足目标 ROI group max overlap `<0.10`，并匹配 token
数量、半球、mean ncsnr 和 parcel 大小。联合条件必须使用 8 或 12 token
的对应对照，不能复用 4-token 对照。

本轮只准运行每类 1 张图、seed 12345 的工程 smoke。smoke 只检查计划、
推理、局部指标和输出审计，不进行正式显著性判断。完成后停止，不自动
启动 10 张 pilot 或全量实验。

Face 局部指标优先使用 E1 OpenCV Haar detector。若运行环境缺少该 API，
smoke 可以显式使用无下载的 bundled LBP fallback，但必须记录 backend，
且 fallback 结果不能直接升级为正式实验。

## D012：E3 pilot 使用严格 Haar 后端并在人工审图后停止

E3 pilot 使用与 E1 完全相同的 OpenCV Haar cascade 与参数。pilot 和
formal 模式禁止 LBP fallback；缺少 `CascadeClassifier` 时必须失败，
不能静默切换 detector。为避免修改共享环境，测试和评价使用项目目录下
的独立 Python 环境。

E3a/E3b 各使用 Face、Body、Scene 每类 10 张冻结图片和 seed 12345。
E3a 每张图 20 个条件，共 600 条记录；E3b 每张图 32 个条件，共 960 条
记录。pilot 只检查工程完整性、效应量、逐图分布和人工视觉结果，不计算
正式 CI、p 或 q。

两项 pilot 完成并通过审计后停止。由于效应方向不一致、逐图异质性较高，
不能从该 pilot 得出类别特异性或分布式冗余的正式结论，也不能自动升级为
三 seed 全量实验。

## D013：E3 正式三 seed 方案与停止规则

E3 正式实验在查看正式结果前冻结 Face 37、Body 50、Scene 50，seeds
`12345/23456/34567`、training-mean replacement、equal-k=4、pure
control overlap `<0.10`、每条件 5 组唯一随机对照，以及全部模型资产、
扩散参数和指标。不得根据 pilot 或正式结果改变这些设置。

E3a 的主要统计先在图像内平均 seed，再使用匹配类别减两个非匹配类别的
等权平均。15 项全局交互统一 BH；27 项局部类别内检验作为独立统计族。

E3b 的联合干预全局 75 项和局部 45 项分别 BH。趋势固定为匹配单 ROI
4 parcels、两个包含匹配 ROI 的双条件之平均 8 parcels、三 ROI 联合
12 parcels，并对每张图片计算最小二乘斜率；全局 15 项和局部 9 项趋势
分别 BH。不得在看到结果后改变 level 顺序、筛选指标或改成单侧检验。

全量解码、统计、强审计和人工审图完成后停止。无论结果方向如何，本轮不
训练新模型、不追加条件，也不把正式数据上的候选局部结果作为同轮追分析
的依据。
