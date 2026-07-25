# ROI 消融实验状态

## 历史性/探索性状态

本目录保留最初对附录 P 的复现尝试。项目现已转向
`experiments/E0_mapping/` 下基于 Algonauts ROI 的公开、可独立复现实验
协议。下述映射差异仍有科研意义，但不再阻塞新的研究。

已有的 50 样本 zero-mask 输出和指标仅属于探索性结果，不得将其作为
最终的功能 ROI 因果证据。

## 映射准入条件：未通过

论文附录 P 要求使用严格的 `>50%` parcel 重叠规则，并报告 Subject 1
有 50 个低层、53 个高层、共 103 个已标注 token。当前 checkpoint 的
parcel 选择已经得到独立验证：根据 NSD 官方 `lh/rh.ncsnr.mgh` 重新计算
parcel 平均 ncsnr，可以在左右半球分别精确复现全部 100 个入选 parcel
索引。

使用官方 Subject 1 和 fsaverage `sphere.reg` 文件，对 NSD 官方原生 ROI
标签进行重采样。对上述已经验证的 200 个 token，得到以下映射：

| 分组 | Token 数 |
|---|---:|
| 低层（V1-V4） | 30 |
| 高层（Face、Body、Scene、Word） | 77 |
| 已标注总数 | 107 |
| 未标注 | 93 |

该结果无法复现论文的 50/53/103 组成。旧的镜像 fsaverage ROI 数组也不
等同于官方映射：与直接使用官方文件重采样的结果相比，其 fLoc 标签存在
显著差异。

## Algonauts 数据来源检查

公开的 NeuroAdapter 数据处理流程采用 Algonauts Project 2023 的
ROI-mask 约定。从该数据集中下载 Subject 1 的 5 类公开 ROI mask 和
5 份代码表，并与本次复现最初使用的镜像文件逐字节比较；全部 15 个文件
完全一致。使用这些 Algonauts masks 重新执行同样的 `>50%` 映射，得到
30 个低层、67 个高层、共 97 个已标注 token。因此，差异不是由使用了
错误的公开 ROI-mask 版本造成的。

## Parcel 来源审计结果

训练 checkpoint 使用
`/public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer`。其中左右
半球 parcel 文件在空间上都是连贯的标准 fsaverage Schaefer 标签。
公开 `whole_brain_encoder` 仓库中提交的 parcel 文件不能被当作忠实于
作者实验的替代参考：公开的右半球 partition 与左半球 partition 完全
相同。当前使用的左半球 partition 与公开左半球文件包含相同的顶点集合，
但索引顺序不同；当前使用的右半球 partition 则与公开右半球文件不一致。

可复现证据位于
`mapping/parcel_provenance_audit_subj01.json`、
`mapping/official_resampled_subj01_nsd_fsaverage_reg/` 和
`mapping/official_mapping_subj01_nsd_fsaverage_reg/`。

## 影响

50 样本 zero-mask pilot 及其官方指标是在执行此准入检查之前完成的。
它们只能作为探索性产物，不得报告为忠实复现论文的功能 ROI 结果。尚未
执行的 mean-mask 和随机匹配对照已在生成输出前停止。

## 必要的下一步

需要获得作者实际使用的 `metadata_sub-01.npy` ROI-mask 字段
（`lh_rois`/`rh_rois`），或者能生成论文 50/53/103 映射的权威预处理
流程。公开代码可以确认实验使用了这些未公开的 metadata 字段，但并未
提供相关数据。只有解决该映射差异后，才能恢复忠实于论文的消融实验。
