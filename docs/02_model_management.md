# 02 Model Management

模型管理建议遵循以下规则：

1. 模型配置和模型权重路径分开。
2. 每个模型使用一个 YAML 描述基本元信息。
3. 本地路径、量化格式、tokenizer 路径都写入配置。
4. 模型实际文件不进入 Git，只在 `manifests/models/` 记录来源与版本。

参考配置：

- [configs/models/qwen2.5-7b-instruct.yaml](/home/jeff-wang/kv-cache-offload-lab/configs/models/qwen2.5-7b-instruct.yaml)
- [configs/models/qwen2.5-14b-instruct.yaml](/home/jeff-wang/kv-cache-offload-lab/configs/models/qwen2.5-14b-instruct.yaml)
- [configs/models/template_model.yaml](/home/jeff-wang/kv-cache-offload-lab/configs/models/template_model.yaml)
