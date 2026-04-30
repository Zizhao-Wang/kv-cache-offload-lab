#!/bin/bash
#SBATCH -J vllm_smoke_test
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o results/%x_%j.out

set -euo pipefail

source env/activate_compute_vllm061.sh
bash scripts/cluster_a100/run_smoke_test.sh
