#!/bin/sh
# Can hand tuning close the gap to the oracle values? Grid over the false-alert weight and
# the attack-flow floor, for the alert formula and the learned estimate (trained without
# the evaluated condition); seeds 2-6. Needs artifacts/models from scripts/learned_eval.sh.
#   caffeinate -i sh scripts/hand_tuning.sh > artifacts/hand-tuning.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
echo "=== grid ($(date '+%H:%M'))"
python scripts/robustness.py --grid --variants base load-x5 separation-5 drift \
  --policies ps-anneal-alerts ps-anneal-learned-loo --seeds 2 3 4 5 6 --workers 2 --resume \
  --output artifacts/hand-tuning
echo "=== done ($(date '+%H:%M'))"
