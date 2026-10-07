#!/bin/bash
# v7: exclude only NEG_* from training (as v4) but apply all weight adjustments incl. hot-visit bins.
# Two seeds to gauge run-to-run scatter; then compare with v4 and v6.
set -e
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy
uv run python scripts/train_ae.py --out outputs/ae_v7 --exclude NEG_BLUE,NEG_NIR --seed 0 > outputs/ae_v7.log 2>&1
uv run python scripts/train_ae.py --out outputs/ae_v7_s1 --exclude NEG_BLUE,NEG_NIR --seed 1 > outputs/ae_v7_s1.log 2>&1
uv run python scripts/compare.py --ae outputs/ae_v4/ae_results.parquet outputs/ae_v6/ae_results.parquet \
    outputs/ae_v7/ae_results.parquet outputs/ae_v7_s1/ae_results.parquet > outputs/compare_v4_v6_v7.txt 2>&1
echo "all finished $(date)" >> outputs/compare_v4_v6_v7.txt
