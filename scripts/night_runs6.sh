#!/bin/sh
# Sixth overnight batch: the adaptive alert formula (L1+L2) on every condition and family,
# and decision-focused weights with feature set v2 on the families (trained on all
# families and once without each), seeds 2-6. Needs artifacts/models/value-gbm-v2*.pkl
# and the family value data from scripts/night_runs5.sh.
#   caffeinate -i sh scripts/night_runs6.sh > artifacts/night-runs6.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
FAMILIES="f1-detectors f2-botnet f3-evasive f4-surges f5-infra"
step() { echo; echo "=== $1 ($(date '+%H:%M'))"; }

step "A. adaptive alert formula"
python scripts/robustness.py --variants $FAMILIES --policies ps-anneal-adaptive ps-anneal-adaptive@fa_weight=0.001 \
  --no-ablations --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/families
python scripts/robustness.py --variants base load-x1 load-x5 separation-5 drift \
  --policies ps-anneal-adaptive ps-anneal-adaptive@fa_weight=0.001 \
  --no-ablations --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/robustness

step "B. decision-focused weights, feature set v2, on the families"
DFL="python scripts/train_dfl.py --features v2 --variants $FAMILIES"
$DFL > artifacts/train-dfl2.log 2>&1 &
$DFL --exclude f1-detectors > artifacts/train-dfl2-without-f1-detectors.log 2>&1
wait
$DFL --exclude f2-botnet > artifacts/train-dfl2-without-f2-botnet.log 2>&1 &
$DFL --exclude f3-evasive > artifacts/train-dfl2-without-f3-evasive.log 2>&1
wait
$DFL --exclude f4-surges > artifacts/train-dfl2-without-f4-surges.log 2>&1 &
$DFL --exclude f5-infra > artifacts/train-dfl2-without-f5-infra.log 2>&1
wait
for f in artifacts/train-dfl2*.log; do echo "--- $f"; grep -v "^weights" "$f" | tail -n 6; done

step "C. evaluate, seeds 2-6"
python scripts/robustness.py --variants $FAMILIES --policies ps-anneal-dfl2 ps-anneal-dfl2-loo \
  --no-ablations --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/families
step "done"
