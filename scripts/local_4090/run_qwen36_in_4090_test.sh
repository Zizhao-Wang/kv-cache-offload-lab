#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

export LAB_ROOT=$HOME/vllm_lab

# Cache / temp dirs
export HF_HOME=$LAB_ROOT/hf_cache
export HF_HUB_CACHE=$LAB_ROOT/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=$LAB_ROOT/hf_cache/hub
export TRANSFORMERS_CACHE=$LAB_ROOT/hf_cache/transformers

export VLLM_CACHE_ROOT=$LAB_ROOT/vllm_cache
export TORCH_HOME=$LAB_ROOT/torch_cache
export TMPDIR=$LAB_ROOT/tmp
export KV_OFFLOAD_DIR=$LAB_ROOT/kv_cache_offload

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$TMPDIR" "$KV_OFFLOAD_DIR"

# 当前没有 nvcc，所以先禁用 FlashInfer sampler
export VLLM_USE_FLASHINFER_SAMPLER=1
unset VLLM_LAB

export PYTHON=$LAB_ROOT/envs/vllm-dev/bin/python

echo "[INFO] python=$PYTHON"
$PYTHON --version
echo "[INFO] VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"

CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" run_qwen36_in_4090_test.py