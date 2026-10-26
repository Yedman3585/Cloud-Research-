#!/bin/sh
# Fourth overnight batch: the learned estimate and the decision-focused weights in iFogSim,
# x3 and drift, seeds 2-6 and 27-41 (disjoint from the training seeds 7-26); then with the
# measured decision time charged (10 ms annealing budget). Every step resumes.
# Needs artifacts/models from scripts/learned_eval.sh and scripts/dfl_eval.sh.
#   caffeinate -i sh scripts/night_runs4.sh > artifacts/night-runs4.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
X3=configs/ids-place-smokescreen-x3.json
SEEDS="2 3 4 5 6 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41"
POLICIES="ps-anneal-alerts ps-anneal-alerts@fa_weight=0.01 ps-anneal-learned-loo@fa_weight=0.001 ps-anneal-dfl-loo ps-anneal-oracle-value"
step() { echo; echo "=== $1 ($(date '+%H:%M'))"; }

for V in base drift; do
  step "A. iFogSim, $V, 20 seeds"
  python scripts/smokescreen_ifogsim.py --config $X3 --variant $V --seeds $SEEDS --policies $POLICIES \
    --workers 2 --resume --output artifacts/learning-ifogsim
done

step "B. iFogSim, x3, measured decision time charged, 10 seeds"
python scripts/smokescreen_ifogsim.py --config $X3 --seeds 2 3 4 5 6 27 28 29 30 31 --decision-time measured \
  --policies ps-anneal-alerts-10ms "ps-anneal-learned-loo@fa_weight=0.001,budget_s=0.01" "ps-anneal-dfl-loo@budget_s=0.01" \
  --workers 2 --resume --output artifacts/learning-ifogsim
step "done"
