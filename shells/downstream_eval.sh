#!/bin/bash
set -euo pipefail

cd /vol/miltank/users/swek/pruna-kvpress-eval
uv sync
source .venv/bin/activate

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config

python scripts/downstream_eval.py --output results/downstream_eval.json
