**项目 A：Collective Evidence Lab——拓扑感知的通信资格验证**

问题：同为 all-reduce，消息大小、rank 放置、缓冲区注册和网络路径不同，性能可能完全不同。单张 `busbw` 截图无法证明原因。

范围：基于 nccl-tests 做可复现研究，不重写测试框架。v1 一台两卡机器，all-reduce 与 all-gather 两种操作；v2 才加入两台物理机器及实际网络。CPU 多进程只做控制与结果解析，不冒充双卡实验。

结构：硬件/软件 manifest → rank 与 CPU/GPU 映射 → 最小链路基线 → collective sweep → 正确性与异常收集 → 原始 JSON/文本 → 分析报告。保留拓扑、实际选用 transport、驱动和 NCCL commit；对工具不支持的输出字段，显式标缺失。

阶段与验收：

1. 先验证单卡内存、GPU 对之间链路和网络状态，再运行集体通信；将启动失败、超时、数值错误与性能低分开。
2. 扫描小消息延迟区与大消息带宽区，覆盖 in-place/out-of-place、不同 rank 排布。基线使用默认设置；一次仅改变一个配置并说明假设。
3. 记录各 rank 与重复间分布、正确性结果、预热和计时语义。NCCL Tests 的 `algbw=S/t` 与 `busbw` 含义不同；all-reduce 的后者使用 `2(n−1)/n` 因子，是操作归一化指标，不能直接当作每条物理链路的遥测值。[指标定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
4. 加入受控的 CPU 亲和性变化、低速/错误接口或独立实验环境中的进程退出，验证检测与超时退出；不在共享生产网络实施故障。
5. 形成一份根因报告：链路瓶颈、放置问题、协议设置或待上游分析。不能因强制某个算法改善单个 case 就建议全局永久固定。

交付：拓扑图、运行命令与环境、原始结果、正确性和失败分类、至少一个可验证性能假设。规划 50–90 小时；CPU 阶段可完成 harness/解析/统计，双卡阶段先预约 4–8 节点小时（即 8–16 GPU 小时），多节点再单独预算。节点、GPU 数、NIC 和互联型号均为预算条件，不假设存在 NVLink。[NCCL 性能排查](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/performance_and_tuning.html)

