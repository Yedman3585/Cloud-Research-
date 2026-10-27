#!/bin/sh
# Seventh overnight batch: the families where the value estimates differ most (F1, F3, F4)
# in iFogSim, seeds 2-6 and 27-41 (disjoint from the training seeds 7-16). Learned and
# decision-focused estimates trained without the evaluated family. Every step resumes.
#   caffeinate -i sh scripts/night_runs7.sh > artifacts/night-runs7.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
X3=configs/ids-place-smokescreen-x3.json
SEEDS="2 3 4 5 6 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41"
POLICIES="ps-anneal-alerts ps-anneal-alerts@fa_weight=0.01 ps-anneal-adaptive ps-anneal-learned2-loo@fa_weight=0.001 ps-anneal-dfl2-loo ps-anneal-oracle-value"
step() { echo; echo "=== $1 ($(date '+%H:%M'))"; }

for V in f1-detectors f4-surges f3-evasive; do
  step "iFogSim, $V, 20 seeds"
  python scripts/smokescreen_ifogsim.py --config $X3 --variant $V --seeds $SEEDS --policies $POLICIES \
    --workers 2 --resume --output artifacts/families-ifogsim
done
step "done"
