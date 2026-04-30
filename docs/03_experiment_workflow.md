# 03 Experiment Workflow

推荐工作流：

1. 选定机器 profile。
2. 选定模型配置和运行时配置。
3. 编写实验 YAML。
4. 执行 smoke test。
5. 再运行正式 KV offload 实验。
6. 将运行元数据写入 `manifests/runs/`，将摘要写入 `results/summaries/`。

关键配置入口：

- [configs/experiments/smoke_test.yaml](/home/jeff-wang/kv-cache-offload-lab/configs/experiments/smoke_test.yaml)
- [configs/experiments/kv_offload_basic.yaml](/home/jeff-wang/kv-cache-offload-lab/configs/experiments/kv_offload_basic.yaml)
- [configs/runtime/vllm061.yaml](/home/jeff-wang/kv-cache-offload-lab/configs/runtime/vllm061.yaml)
