#!/bin/sh
# Third overnight batch: training data on more conditions, drift on 20 seeds in iFogSim,
# oracle values in iFogSim; every step resumes.
#   caffeinate -i sh scripts/night_runs3.sh > artifacts/night-runs3.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
X3=configs/ids-place-smokescreen-x3.json
SEEDS20="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21"
step() { echo; echo "=== $1 ($(date '+%H:%M'))"; }

step "A. value data on other conditions, training seeds 7-16"
for V in load-x1 load-x5 separation-5 drift; do
  python scripts/collect_values.py --variant $V --seeds 7 8 9 10 11 12 13 14 15 16 --workers 2 --resume
done

step "B. iFogSim, non-stationary scenario, 20 seeds, with oracle values"
python scripts/smokescreen_ifogsim.py --config $X3 --variant drift --seeds $SEEDS20 \
  --policies greedy-v2 value-greedy-alerts ps-anneal-alerts sa-toca sa-toca-own ps-anneal-oracle-value \
  --workers 2 --resume --output artifacts/robustness-ifogsim

step "C. iFogSim, x3, 20 seeds: oracle values"
python scripts/smokescreen_ifogsim.py --config $X3 --seeds $SEEDS20 --policies ps-anneal-oracle-value \
  --workers 2 --resume --output artifacts/smokescreen-ifogsim
step "done"
