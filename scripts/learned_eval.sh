#!/bin/sh
# First test of the learned value estimate (two-stage): train on the value data of every
# condition (and once without each condition), then evaluate in the Python simulator.
#   caffeinate -i sh scripts/learned_eval.sh > artifacts/learned-eval.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
step() { echo; echo "=== $1 ($(date '+%H:%M'))"; }

step "A. train (all conditions, and leave one condition out)"
python scripts/train_values.py --loo || exit 1

step "B. evaluate, seeds 2-6"
python scripts/robustness.py --variants base load-x1 load-x5 separation-5 drift \
  --policies ps-anneal-learned ps-anneal-learned-loo ps-anneal-true-a --no-ablations \
  --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/robustness
step "done"
