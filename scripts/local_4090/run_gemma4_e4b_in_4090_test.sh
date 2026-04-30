#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

export LAB_ROOT=$HOME/vllm_lab

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

export VLLM_USE_FLASHINFER_SAMPLER=1
unset VLLM_LAB
unset VLLM_ATTENTION_BACKEND

export PYTHON=$LAB_ROOT/envs/vllm-dev/bin/python

echo "[INFO] python=$PYTHON"
$PYTHON --version
echo "[INFO] VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" run_gemma4_e4b_in_4090_test.py
