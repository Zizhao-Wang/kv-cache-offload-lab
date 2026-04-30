# KV Cache Offload Lab

用于管理 KV cache offloading 研究的仓库骨架，覆盖以下几类内容：

- `configs/`: 机器、模型、实验、运行时配置
- `docs/`: 环境说明、工作流、迁移与结果管理文档
- `env/`: 环境激活、检查、导出、打包脚本
- `scripts/`: 本地运行、集群运行、日志分析辅助脚本
- `slurm/`: A100 集群任务模板
- `manifests/`: 环境、模型、运行记录清单
- `results/`: 汇总结果、图表、表格

当前保留了两份原始研究文档：

- [docs/kv_cache_offload_research_workflow.md](/home/jeff-wang/kv-cache-offload-lab/docs/kv_cache_offload_research_workflow.md)
- [docs/vllm_offline_setup_and_model_plan.md](/home/jeff-wang/kv-cache-offload-lab/docs/vllm_offline_setup_and_model_plan.md)

建议优先从 [docs/00_environment.md](/home/jeff-wang/kv-cache-offload-lab/docs/00_environment.md) 开始阅读。
