#!/bin/bash
#SBATCH -J kv_offload_basic
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -o results/%x_%j.out

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/ssdcache/zizhaowang/vllm_lab/models/Qwen2.5-7B-Instruct}"

source env/activate_compute_vllm061.sh
bash scripts/cluster_a100/run_vllm_compute.sh "${MODEL_PATH}"
