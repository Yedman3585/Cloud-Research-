#!/bin/sh
# Fifth overnight batch: the scenario families of docs/learning-plan.md.
# A: headroom per family (formula, tuned formula, true attack flows, oracle values);
# B: value data per family (training seeds 7-16); C: attack-flow models with feature set
# v2, trained on all families and once without each; D: their evaluation, seeds 2-6.
#   caffeinate -i sh scripts/night_runs5.sh > artifacts/night-runs5.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
FAMILIES="f1-detectors f2-botnet f3-evasive f4-surges f5-infra"
step() { echo; echo "=== $1 ($(date '+%H:%M'))"; }

step "A. headroom per family, seeds 2-6"
python scripts/robustness.py --variants $FAMILIES \
  --policies greedy-v2 ps-anneal-alerts ps-anneal-alerts@fa_weight=0.01 ps-anneal-true-a ps-anneal-oracle-value \
  --no-ablations --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/families

step "B. value data per family, seeds 7-16"
for V in $FAMILIES; do
  python scripts/collect_values.py --variant $V --seeds 7 8 9 10 11 12 13 14 15 16 --workers 2 --resume
done

step "C. attack-flow models, feature set v2"
python scripts/train_values.py --variants $FAMILIES --features v2 --loo --prefix value-gbm-v2 || exit 1

step "D. learned estimate on the families, seeds 2-6"
python scripts/robustness.py --variants $FAMILIES \
  --policies ps-anneal-learned2@fa_weight=0.001 ps-anneal-learned2-loo@fa_weight=0.001 \
  --no-ablations --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/families
step "done"
