#!/usr/bin/env bash
set -euo pipefail

echo "========== Basic Info =========="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "PWD : $(pwd)"

echo
echo "========== OS =========="
cat /etc/os-release || true
uname -a || true

echo
echo "========== GPU =========="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "[WARN] nvidia-smi not found"
fi

echo
echo "========== Python =========="
which python || true
python --version || true

echo
echo "========== Python Packages =========="
python - <<'PY'
import importlib.util

for pkg in ["torch", "vllm", "transformers", "huggingface_hub"]:
    spec = importlib.util.find_spec(pkg)
    if spec is None:
        print(f"[MISSING] {pkg}")
    else:
        print(f"[OK] {pkg}")

try:
    import torch
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"gpu {i}:", torch.cuda.get_device_name(i))
except Exception as e:
    print("[WARN] torch check failed:", repr(e))

try:
    import vllm
    print("vllm version:", vllm.__version__)
except Exception as e:
    print("[WARN] vllm check failed:", repr(e))
PY

echo
echo "========== Disk =========="
df -h .


