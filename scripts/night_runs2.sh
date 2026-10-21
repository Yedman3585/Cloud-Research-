#!/bin/sh
# Second overnight batch: literature baselines and adaptivity; every step resumes.
#   caffeinate -i sh scripts/night_runs2.sh > artifacts/night-runs2.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
X3=configs/ids-place-smokescreen-x3.json
SEEDS20="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21"
step() { echo; echo "=== $1 ($(date '+%H:%M'))"; }

step "A. tune SA-TOCA on each condition (training seed 7), three in parallel"
for V in load-x2 extract-1us drift; do
  if [ ! -f configs/sa-toca-x3-$V.json ]; then
    python scripts/tune_satoca.py --seeds 7 --iterations 600 --variant $V > artifacts/tune-satoca-$V.log 2>&1 &
  fi
done
wait
tail -n 1 artifacts/tune-satoca-*.log

step "B. Python simulator, seeds 2-6: static splits tuned at x3 and on each condition"
python scripts/robustness.py --variants load-x2 extract-1us --policies sa-toca sa-toca-own --no-ablations \
  --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/baselines
python scripts/robustness.py --variants drift --policies greedy-v2 value-greedy-alerts ps-anneal-alerts sa-toca sa-toca-own \
  --no-ablations --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/baselines
python scripts/robustness.py --variants base --policies sa-toca fog-qiea qiea-ps --no-ablations \
  --seeds 2 3 4 5 6 --workers 2 --resume --output artifacts/baselines

step "C. iFogSim, x3, 20 seeds: SA-TOCA and FOG-QIEA"
python scripts/smokescreen_ifogsim.py --config $X3 --seeds $SEEDS20 --policies sa-toca fog-qiea \
  --workers 2 --resume --output artifacts/smokescreen-ifogsim

step "D. iFogSim, non-stationary scenario, seeds 2-6"
python scripts/smokescreen_ifogsim.py --config $X3 --variant drift --seeds 2 3 4 5 6 \
  --policies greedy-v2 value-greedy-alerts ps-anneal-alerts sa-toca sa-toca-own \
  --workers 2 --resume --output artifacts/robustness-ifogsim
step "done"
