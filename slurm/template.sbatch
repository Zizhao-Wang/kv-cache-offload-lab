#!/bin/bash
#SBATCH -J template_job
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o results/%x_%j.out

set -euo pipefail

echo "replace with your job body"
