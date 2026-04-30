#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <conda_env_name> <output_tar_gz>" >&2
  exit 1
fi

ENV_NAME="$1"
OUT_FILE="$2"

conda pack -n "${ENV_NAME}" -o "${OUT_FILE}"
echo "packed ${ENV_NAME} -> ${OUT_FILE}"
