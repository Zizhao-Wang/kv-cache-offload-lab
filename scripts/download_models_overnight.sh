#!/bin/bash
set -euo pipefail

export VLLM_LAB=$HOME/vllm_lab
export HF_HOME=$VLLM_LAB/hf_cache
export HF_HUB_CACHE=$VLLM_LAB/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=$VLLM_LAB/hf_cache/hub

source "$VLLM_LAB/envs/hf-download/bin/activate"

mkdir -p "$VLLM_LAB/models" "$VLLM_LAB/logs" "$HF_HUB_CACHE"

echo "[$(date)] Start downloading Qwen3.6-27B..."
nohup hf download Qwen/Qwen3.6-27B \
  --local-dir "$VLLM_LAB/models/Qwen3.6-27B" \
  --max-workers 8 \
  > "$VLLM_LAB/logs/download_qwen3.6_27b.log" 2>&1 &
echo $! > "$VLLM_LAB/logs/download_qwen3.6_27b.pid"

echo "[$(date)] Start downloading Gemma-4-E4B-it..."
nohup hf download google/gemma-4-E4B-it \
  --local-dir "$VLLM_LAB/models/gemma-4-E4B-it" \
  --max-workers 8 \
  > "$VLLM_LAB/logs/download_gemma4_e4b_it.log" 2>&1 &
echo $! > "$VLLM_LAB/logs/download_gemma4_e4b_it.pid"

echo "Download jobs started:"
echo "Qwen PID:  $(cat "$VLLM_LAB/logs/download_qwen3.6_27b.pid")"
echo "Gemma PID: $(cat "$VLLM_LAB/logs/download_gemma4_e4b_it.pid")"
echo
echo "Check logs with:"
echo "tail -f $VLLM_LAB/logs/download_qwen3.6_27b.log"
echo "tail -f $VLLM_LAB/logs/download_gemma4_e4b_it.log"
