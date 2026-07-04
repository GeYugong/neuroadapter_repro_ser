# 当前进度

## 已完成

- SSH 可通过 `A40_Cluster_1` 连接。
- 个人目录已建立：`/public/home/mty/GeYugong`。
- NeuroAdapter 代码已放到：`/public/home/mty/GeYugong/code/NeuroAdapter`。
- Conda 环境已建立：`neuroadapter`。
- 核心 fake-data 模型测试已跑通。

## 服务器硬件

```text
6 x NVIDIA A40，单卡约 46GB 显存
```

## 下一步

- 向师兄确认 NSD 数据、parcel 文件、whole_brain_encoder 权重、Stable Diffusion 权重路径。
- 把训练脚本中的硬编码路径改成配置参数或本仓库 configs 文件。
- 先跑 subject 1 的小规模 dataloader / train smoke test。
