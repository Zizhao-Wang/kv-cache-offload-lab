# 00 Environment

本仓库默认面向两类环境：

- 本地单卡开发机，例如 `RTX 4090`
- 集群离线计算节点，例如 `comput10` 上的 `A100`

核心原则：

1. 联网下载、打包、同步在管理节点完成。
2. 计算节点只负责离线运行与结果产出。
3. 模型、缓存、运行日志与结果目录分离管理。
4. 所有实验尽量由配置文件和脚本驱动，减少临时命令。

相关文档：

- [01_machine_profiles.md](/home/jeff-wang/kv-cache-offload-lab/docs/01_machine_profiles.md)
- [02_model_management.md](/home/jeff-wang/kv-cache-offload-lab/docs/02_model_management.md)
- [03_experiment_workflow.md](/home/jeff-wang/kv-cache-offload-lab/docs/03_experiment_workflow.md)
- [04_result_management.md](/home/jeff-wang/kv-cache-offload-lab/docs/04_result_management.md)
- [05_migration_4090_to_a100.md](/home/jeff-wang/kv-cache-offload-lab/docs/05_migration_4090_to_a100.md)

已有详细背景记录：

- [kv_cache_offload_research_workflow.md](/home/jeff-wang/kv-cache-offload-lab/docs/kv_cache_offload_research_workflow.md)
- [vllm_offline_setup_and_model_plan.md](/home/jeff-wang/kv-cache-offload-lab/docs/vllm_offline_setup_and_model_plan.md)
