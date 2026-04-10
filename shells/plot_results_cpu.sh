#!/bin/bash
set -euo pipefail

cd /vol/miltank/users/swek/pruna-kvpress-eval
uv sync
source .venv/bin/activate

source shells/_machine_config.sh
validate_config

python scripts/plot_results.py
