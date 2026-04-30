#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import sys

print("python_executable:", sys.executable)

try:
    import torch
    print("torch_version:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_name:", torch.cuda.get_device_name(0))
        print("gpu_count:", torch.cuda.device_count())
except Exception as exc:
    print("torch_check_failed:", exc)

try:
    import vllm
    print("vllm_version:", vllm.__version__)
except Exception as exc:
    print("vllm_check_failed:", exc)
PY
