# 面试与简历证据

当前可用表述：

> Built a reproducible NCCL qualification harness with topology manifests, native per-iteration evidence, correctness gates and process-group timeout cleanup; validated 30 real single-GPU runs and detected zero-byte all-gather cases caused by upstream size rounding.

可以现场演示：运行 Python 回归；展示旧的零字节原生 JSON 与拒绝原因；解释 `S/t` 和两个 bus factor；展示两卡请求在单卡环境下被阻断；追溯任意一条统计到原始数据和哈希。

需要讲清的技术选择：为何使用 nccl-tests 原生 JSON、为何总 gathered size 不等于每 rank 输入、为何 run-average 与逐次 CUDA event 时间不能混成同一种分布、为何拓扑工具不等于实际传输证据。

暂不可写：双卡通信优化幅度、NVLink/网络带宽、跨节点 RDMA 调优或已经完成拓扑根因定位。接入双卡环境并取得结果后，再把具体硬件、配置差异和实测幅度补入简历。
