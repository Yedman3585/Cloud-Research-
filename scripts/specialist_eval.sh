#!/usr/bin/env bash
# Evaluate specialist DFL models against adaptive and general DFL on each family.
# Policies: ps-anneal-alerts (formula), ps-anneal-adaptive, ps-anneal-dfl2-loo,
#           ps-anneal-dfl2-specialist, ps-anneal-oracle-value.
# Seeds 2-6, families only.
#
#   bash scripts/specialist_eval.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="2 3 4 5 6"
FAMILIES="f1-detectors f2-botnet f3-evasive f4-surges f5-infra"
POLICIES="ps-anneal-alerts ps-anneal-adaptive ps-anneal-dfl2-loo ps-anneal-dfl2-specialist ps-anneal-oracle-value"
OUT="artifacts/specialist-eval"

mkdir -p "$OUT"

echo "=== Specialist evaluation: families x policies, seeds $SEEDS ==="
python scripts/robustness.py \
    --output "$OUT" \
    --policies $POLICIES \
    --variants $FAMILIES \
    --seeds $SEEDS \
    --workers 2 \
    --resume \
    2>&1 | tee "$OUT/eval.log"

echo ""
echo "=== Done. Results in $OUT/runs.jsonl ==="
