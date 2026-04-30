#!/bin/bash

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

# CUDA Toolkit installed inside micromamba env, not system-wide.
export CUDA_HOME=$LAB_ROOT/envs/vllm-dev
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Add CUDA headers and libs to env vars so that nvcc can find them.
export NVIDIA_CUDA_PYPI_ROOT="$CUDA_HOME/lib/python3.12/site-packages/nvidia/cu13"
export CPATH="$NVIDIA_CUDA_PYPI_ROOT/include:$CUDA_HOME/include:${CPATH:-}"
export C_INCLUDE_PATH="$NVIDIA_CUDA_PYPI_ROOT/include:$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$NVIDIA_CUDA_PYPI_ROOT/include:$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$NVIDIA_CUDA_PYPI_ROOT/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NVIDIA_CUDA_PYPI_ROOT/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "NVIDIA_CUDA_PYPI_ROOT=$NVIDIA_CUDA_PYPI_ROOT"
echo "CPATH=$CPATH"
find "$NVIDIA_CUDA_PYPI_ROOT/include" -name curand.h


# Current safe default.
# Disable FlashInfer sampler unless we explicitly test nvcc/JIT path.
export VLLM_USE_FLASHINFER_SAMPLER=0

micromamba activate "$LAB_ROOT/envs/vllm-dev"