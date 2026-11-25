#!/usr/bin/env bash
# Train specialist (family-specific) value and DFL models.
# Each family gets its own GBM attack-flow predictor and DFL weight vector,
# trained only on that family's data (seeds 7-16).
#
#   bash scripts/train_specialists.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

FAMILIES=(f1-detectors f2-botnet f3-evasive f4-surges f5-infra)

for fam in "${FAMILIES[@]}"; do
    echo "=== Training specialist models for $fam ==="

    # 1. Value model (attack-flow regressor) on this family only
    echo "  GBM: value-gbm-v2-only-${fam}"
    python scripts/train_values.py \
        --features v2 \
        --variants "$fam" \
        --prefix "value-gbm-v2-only-${fam}" \
        2>&1 | tail -3

    # 2. DFL weight vector on this family only
    echo "  DFL: dfl2-only-${fam}"
    python scripts/train_dfl.py \
        --features v2 \
        --variants "$fam" \
        --seeds-per-condition 10 \
        --output "artifacts/models/dfl2-only-${fam}.json" \
        2>&1 | tail -5

    echo ""
done

echo "=== All specialist models trained ==="
ls -lh artifacts/models/dfl2-only-*.json artifacts/models/value-gbm-v2-only-*.pkl 2>/dev/null
