# E2 ROI overlap 与随机对照污染审计

本目录逐项解析 `all_roi_overlaps`，而不是只读取 dominant ROI。组级 overlap
取组内具体 atlas ROI 的最大值，不将不同 mask 相加为伪造的并集比例。

正式 plan 包含 59 个目标 parcel，其中 8 个同时对目标组和非目标组达到
0.5 overlap。冻结 full matched controls 中，对目标组 overlap ≥0.5 的
数量为：Face 0、Body 1、Scene 4；≥0.1 的数量分别为 3、14、17。
这些数量是 `category × design × replicate × control parcel` 级别的
control 记录数；同一个 parcel 可能在不同 replicate 中重复出现，因此
不能直接解释为独立 parcel 数量。

Face 的 4 个目标 parcel 可以直观看到纯度问题：token 16 的 Face/Body
overlap 为 `0.832/0.766`，token 22 的 Face/V3 为 `0.554/0.527`，
token 152 的 Face/Word 为 `0.618/0.554`；token 43 的最大非目标 overlap
为 Body `0.329`，相对更纯。

四个排除阈值 `<0.10`、`<0.25`、`<0.40`、`<0.50` 下，三个类别均可
构造 5 组 matched controls。该结果只证明未来 pure-control sensitivity
可行，本阶段没有替换正式 zero/mean 使用的冻结随机对照。更严格阈值会
减少候选池，并可能增大 SNR 与 parcel size 的匹配距离。
