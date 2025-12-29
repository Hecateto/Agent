
# MPCache: A Cache-Aware Framework for Secure Inference in Multi-Party Computation

## Motivation

解决 LLM 安全推理的问题：

- **KV Cache Eviction**：考虑传统 LLM 安全推理与 KV Cache 缓存优化技术的结合，实现安全推理加速，降低时延和通信开销。


## Design

1.  分析Transformer架构（MLP、Attention、LayerNorm）的计算分布，通过"Before After"对比，展示缓存优化前后的性能差异和瓶颈；
2. 基于观察 (稀疏性，MPC计算瓶颈， 相邻层共享等)设计框架步骤和优化手段，包括：`Look-once Static KV Cache Eviction`，`MPC-friendly Dynamic KV Cache Selection`，`Cross-layer Index-sharing Strategy`，`Hierachical Clustering of Key Cache`等，在实现最小性能损失的前提下最优化开销。

## Results

-   提出MPCache框架，显著降低了Transformer模型在MPC环境中的通信开销和推理延迟，适用于不同序列长度（64-4096）
-  与SOTA的各种KV Cache Eviction策略相比，在安全推理的密文态下实现性能和开销最优