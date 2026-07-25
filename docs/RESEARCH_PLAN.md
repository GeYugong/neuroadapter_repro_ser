# 功能脑区因果研究计划

## 研究目标

本项目研究功能性 fMRI 脑区是否会对特定类别的图像重建产生因果贡献，
以及 top-SNR parcel 选择是否会使相关结论产生偏差。

严格复现附录 P 中的 50/53/103 parcel 数量已不再是主要目标。主实验协议
采用公开的 Algonauts Project 2023 Subject 1 fsaverage ROI masks。

## 研究问题

1. 屏蔽与图像类别相匹配的 ROI，相比屏蔽无关 ROI 或匹配的随机 parcel，
   是否会使对应类别刺激的重建效果下降更多？
2. 零值替换和训练集均值替换得到的效应方向是否一致？
3. top-SNR-200 会如何改变功能 ROI 的覆盖率和组成？
4. 核心研究完成后，attention 大小能否预测屏蔽效应？

## 实验阶段

- E0：建立完整的公开 ROI 清单，并量化 top-SNR 选择偏差。
- E1：为 Face、Body、Scene 和探索性 Word 类别建立刺激清单。
- E2：在已有 top-SNR-200 模型上进行配对因果屏蔽实验。
- E3：比较新训练的 top-SNR-200 与 ROI-balanced-200 模型。
- E4：可选的 attention 与因果效应关系分析。

## 当前范围

阶段 A/B、E2 zero-mask pilot 与正式 GPU 解码均已完成。正式实验没有
发现校正显著的正向类别特异性效应，该负结果永久保留。当前进入预注册的
E2b mean-mask 稳健性分析；除将全零替换改为训练集 parcel-wise mean
替换外，样本、checkpoint、parcel、随机对照、seed、扩散参数、指标和
统计方案均保持不变。本阶段不训练模型。

E0 发现所有通过公开数据映射出的功能 ROI 均有 100% 的保留率。因此，
计划中的 ROI-balanced-200 训练对比已暂停：主映射结果不支持原先所假设
的 ROI 覆盖不足问题。
