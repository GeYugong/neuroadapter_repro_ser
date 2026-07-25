# E2 ROI overlap 与随机对照污染审计

本目录逐项解析 `all_roi_overlaps`，而不是只读取 dominant ROI。组级 overlap
取组内具体 atlas ROI 的最大值，不将不同 mask 相加为伪造的并集比例。

正式 plan 包含 59 个目标 parcel，其中 8 个同时对目标组和非目标组达到
0.5 overlap。冻结 full matched controls 中，对目标组 overlap ≥0.5 的
数量为：Face 0、Body 1、Scene 4；≥0.1 的数量分别为 3、14、17。

四个排除阈值 `<0.10`、`<0.25`、`<0.40`、`<0.50` 下，三个类别均可
构造 5 组 matched controls。该结果只证明未来 pure-control sensitivity
可行，本阶段没有替换正式 zero/mean 使用的冻结随机对照。
