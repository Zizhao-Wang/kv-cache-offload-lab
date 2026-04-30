#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="${1:-/ssdcache/zizhaowang/vllm_lab/envs/vllm061-native}"

if [[ ! -f "${ENV_ROOT}/bin/activate" ]]; then
  echo "missing activate script: ${ENV_ROOT}/bin/activate" >&2
  exit 1
fi

source "${ENV_ROOT}/bin/activate"

export HF_HOME="${HF_HOME:-/ssdcache/zizhaowang/vllm_lab/hf_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/ssdcache/zizhaowang/vllm_lab/vllm_cache}"
export TORCH_HOME="${TORCH_HOME:-/ssdcache/zizhaowang/vllm_lab/torch_cache}"

echo "activated: ${ENV_ROOT}"
echo "python: $(command -v python)"
