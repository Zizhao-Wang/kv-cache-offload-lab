# 01 Machine Profiles

本目录将机器差异显式化，避免把路径、GPU 数量、缓存目录写死在脚本里。

建议维护三类 profile：

- `local_4090`: 本地单卡开发与小规模 smoke test
- `cluster_a100_comput10`: 集群离线 A100 节点
- `standalone_a100_future`: 未来可联网或半联网的独立 A100 机器

对应配置文件位于：

- [configs/machines/local_4090.example.env](/home/jeff-wang/kv-cache-offload-lab/configs/machines/local_4090.example.env)
- [configs/machines/cluster_a100_comput10.example.env](/home/jeff-wang/kv-cache-offload-lab/configs/machines/cluster_a100_comput10.example.env)
- [configs/machines/standalone_a100_future.example.env](/home/jeff-wang/kv-cache-offload-lab/configs/machines/standalone_a100_future.example.env)

复制示例后，按实际机器改成不提交到仓库的 `.env` 文件。
