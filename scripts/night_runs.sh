#!/bin/sh
# Overnight runs for the reviewers' questions, most valuable first; every step resumes.
#   caffeinate -i sh scripts/night_runs.sh > artifacts/night-runs.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
X3=configs/ids-place-smokescreen-x3.json
SEEDS20="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21"
POL="greedy-v2 value-greedy-alerts ps-anneal ps-anneal-alerts"
step() { echo; echo "=== $1 ($(date '+%H:%M'))"; }

step "1. main comparison in iFogSim, 20 seeds"
python scripts/smokescreen_ifogsim.py --config $X3 --seeds $SEEDS20 --policies $POL \
  --workers 2 --resume --output artifacts/smokescreen-ifogsim
step "1b. oracle values, Python simulator, 20 seeds"
python scripts/robustness.py --variants base --policies ps-anneal-oracle-value --no-ablations \
  --seeds $SEEDS20 --workers 2 --resume --output artifacts/oracle-20

step "2. decision time charged in iFogSim: measured (one worker, clean timing)"
python scripts/smokescreen_ifogsim.py --config $X3 --seeds 2 3 4 5 6 --policies $POL ps-anneal-alerts-10ms \
  --decision-time measured --workers 1 --resume --output artifacts/decision-time
step "2b. decision time charged in iFogSim: fixed 16 ms and 50 ms"
python scripts/smokescreen_ifogsim.py --config $X3 --seeds 2 3 4 5 6 --policies $POL \
  --decision-time 0.016 0.05 --workers 2 --resume --output artifacts/decision-time

step "5. training data for a learned value estimate, seeds 7-26"
python scripts/collect_values.py --seeds 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 \
  --workers 2 --resume

step "3. load curve, Python simulator"
python scripts/robustness.py --variants load-x1 load-x1.5 load-x2 load-x2.5 base load-x4 load-x5 \
  --no-ablations --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/robustness

step "4. robustness in iFogSim"
for V in extract-1us separation-5 large; do
  python scripts/smokescreen_ifogsim.py --config $X3 --variant $V --seeds 2 3 4 5 6 --policies $POL \
    --workers 2 --resume --output artifacts/robustness-ifogsim
done
step "done"
