#!/bin/sh
# Decision-focused value weights: train on every condition and once without each
# condition (two trainings at a time), then evaluate in the Python simulator, seeds 2-6.
# Needs artifacts/models/value-gbm*.pkl from scripts/learned_eval.sh.
#   caffeinate -i sh scripts/dfl_eval.sh > artifacts/dfl-eval.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
step() { echo; echo "=== $1 ($(date '+%H:%M'))"; }

step "A. train"
python scripts/train_dfl.py > artifacts/train-dfl.log 2>&1 &
python scripts/train_dfl.py --exclude base > artifacts/train-dfl-without-base.log 2>&1
wait
python scripts/train_dfl.py --exclude load-x1 > artifacts/train-dfl-without-load-x1.log 2>&1 &
python scripts/train_dfl.py --exclude load-x5 > artifacts/train-dfl-without-load-x5.log 2>&1
wait
python scripts/train_dfl.py --exclude separation-5 > artifacts/train-dfl-without-separation-5.log 2>&1 &
python scripts/train_dfl.py --exclude drift > artifacts/train-dfl-without-drift.log 2>&1
wait
for f in artifacts/train-dfl*.log; do echo "--- $f"; grep -v "^weights" "$f" | tail -n 12; done

step "B. evaluate, seeds 2-6"
python scripts/robustness.py --variants base load-x1 load-x5 separation-5 drift \
  --policies ps-anneal-dfl ps-anneal-dfl-loo --no-ablations \
  --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/robustness
step "done"
