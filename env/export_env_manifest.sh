#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-manifests/env}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_DIR}/env_manifest_${STAMP}.txt"

mkdir -p "${OUT_DIR}"

{
  echo "timestamp=${STAMP}"
  echo "hostname=$(hostname)"
  echo "python=$(command -v python || true)"
  python --version 2>/dev/null || true
  pip --version 2>/dev/null || true
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
  python - <<'PY'
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
except Exception as exc:
    print(f"torch_error={exc}")

try:
    import vllm
    print(f"vllm={vllm.__version__}")
except Exception as exc:
    print(f"vllm_error={exc}")
PY
} > "${OUT_FILE}"

echo "wrote ${OUT_FILE}"
