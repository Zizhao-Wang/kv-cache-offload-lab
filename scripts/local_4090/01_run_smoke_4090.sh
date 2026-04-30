#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${MODEL_PATH:-}" ]; then
    echo "[ERROR] MODEL_PATH is not set."
    echo
    echo "Please run something like:"
    echo "  export MODEL_PATH=/path/to/your/Qwen2.5-7B-Instruct"
    echo
    exit 1
fi

RUN_DIR="runs/4090_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Run dir  : $RUN_DIR"
echo "[INFO] Model    : $MODEL_PATH"

echo
echo "========== Step 1: Check machine =========="
./scripts/00_check_machine.sh | tee "$RUN_DIR/machine_check.txt"

echo
echo "========== Step 2: Run vLLM smoke test =========="
python scripts/01_vllm_smoke_test.py 2>&1 | tee "$RUN_DIR/vllm_smoke.log"

echo
echo "[OK] Finished."
echo "[OK] Logs saved to: $RUN_DIR"
