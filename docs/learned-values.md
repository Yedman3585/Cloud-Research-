# Learned value coefficients: model, training and results

One place for everything about the learned part of the method: what is learned, how it is
trained, and every result against the hand-made estimates, in the Python simulator and in
iFogSim. Plan and hypotheses (committed before the experiments):
[learning-plan.md](learning-plan.md). Details per experiment: [smokescreen.md](smokescreen.md)
(earlier conditions) and [families.md](families.md) (scenario families).

## 1. What is learned, and why only that

Every epoch the scheduler solves the epoch model (exact processor-sharing finish times,
deadline penalties) by annealing. The model has two kinds of coefficients:

| Coefficients | Source | Learned? |
|---|---|---|
| load terms: seconds of exclusive use of every CPU and uplink per option, budgets to the deadlines | computed exactly from the instance | no (exact; iFogSim agrees within milliseconds) |
| option values: how much analysing a task with a model and sampling fraction lowers the objective | unknown at decision time: whether and how much a task is attacked | **yes** |

The hand-made value of option `o` of task `t` is

    v(t, o) = crit(t) x share(t) x p(o, a(t)) - w x fa(t, o)
    p(o, a) = 1 - (1 - f(o) x recall(model(o)))^a        fa = f(o) x n(t) x (1 - prefilter(t)) x fpr(model(o))

with `a(t)` the estimated attack flows of the task: from the gateway's alerts over 40 s for
residual tasks (`max(3, share_40 x n)`), from the pre-filter for flood tasks. `w` is the
false-alert weight (1e-4 as set on x3).

The learned estimates replace, step by step:

1. **`a(t)`** by a regression model (*learned*, two-stage);
2. **the two weights** (`crit x share` scale and `w`) by task-dependent weights trained on
   decision quality (*DFL*).

The annealer, the budget, the repair and the load terms stay the same in every policy.

## 2. The learned attack-flow estimate (two-stage)

Code: [`fogids/learned.py`](../src/fogids/learned.py),
[`scripts/train_values.py`](../scripts/train_values.py),
[`scripts/collect_values.py`](../scripts/collect_values.py).

- **Model:** gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`, Poisson
  loss, 400 iterations, 31 leaves). Prediction: about 3 ms per epoch.
- **Target:** the attack flows of the task that the value stands for: low-rate flows in a
  residual task (flood flows leaked there by the header pass are left out), all attack flows
  in a flood task.
- **Features, set v1** (21): stream, pre-filter score, flows, packets, criticality, deadline;
  the gateway's alerts, expected false alerts, analysed recall, excess share and z-score
  over 10, 20 and 40 s.
- **Features, set v2** (+9, scenario families): bytes per packet; the gateway's device
  profile (camera, hub, sensor: device types are known to the operator); alerts, expected
  false alerts, analysed recall and excess share of the other gateways of the same fog
  subtree over 40 s.
- **Data:** every task the scheduler saw in runs of `ps-anneal-alerts` (features at decision
  time, ground truth afterwards). Seeds 7–26 (x3) and 7–16 (every other condition and
  family); 538 000 tasks on the earlier conditions, 441 000 on the families.
- **Evaluation:** seeds 2–6 (Python) and 2–6 plus 27–41 (iFogSim), disjoint from training.
  *Unseen* (`-loo`): trained without the evaluated condition or family.
- **Floor.** The estimate is floored at 3 flows, like the formula. Without it the learned
  estimate stops analysing tasks that have no alerts yet, which then never raise any
  (objective 0.71 on one x3 seed): the estimate shapes its own future features.

Prediction quality (rank correlation with the true low-rate flows of residual tasks,
held-out seeds): 0.79–0.84 for the model against 0.43–0.73 for the alert formula on the
earlier conditions; on the families 0.86 (F1), 0.60 (F2), 0.06 (F3), 0.90 (F4), 0.91 (F5)
for the model against 0.08, 0.39, −0.03, 0.51, 0.50 for the formula.

## 3. Decision-focused weights (DFL)

Code: [`fogids/dfl.py`](../src/fogids/dfl.py), [`scripts/train_dfl.py`](../scripts/train_dfl.py).

    v(t, o) = s(t) x crit(t) x share(t) x p(o, a(t)) - lam(t) x fa(t, o)
    s(t) = exp(theta_s . phi(t)),  lam(t) = exp(theta_l . phi(t))

`phi(t)`: 16 task features (21 with set v2), standardised; 32 (42) parameters. Start:
`s = 1`, `lam = 1e-3` (the best hand setting of the learned estimate).

- **Training signal:** on recorded epoch models (every option with its loads and budgets,
  the exact value of every option from ground truth), the regret of the annealer's solution
  under the predicted values, measured with the exact values against the annealer's
  solution under the exact values (averaged over three solver seeds).
- **SPO+ did not work:** its subgradient first changes the overall scale of the values, and
  here the scale is not free: deadline penalties and the jobs already committed are in
  absolute units. Validation regret rose from the first pass on.
- **Used instead: zeroth-order descent on the regret** — antithetic Gaussian perturbations
  of theta (8 pairs, sigma 0.1), 150 random epochs per step, 80 steps, normalised steps with
  decaying size, L2 pull to the start, early stopping on the validation regret (last
  training seed of every condition). Lowers the validation regret by about 10 %.
- **Data:** 6 seeds per condition, every third epoch (about 5 000 epoch models); the
  attack-flow estimate in the training data is cross-fitted (trees trained on the other
  seed parity), so that it is never evaluated on its own training data.

## 4. Competitors without learning

- **formula** (`ps-anneal-alerts`), false-alert weight 1e-4; **formula, weight 1e-2**: the
  hand-tuned weight that was best for the formula on x5, drift and 5 % separation error
  (grid over 1e-4 … 3e-2 and floors 3 and 1, best chosen on the evaluation seeds);
- **adaptive** ([`fogids/adaptive.py`](../src/fogids/adaptive.py)): the formula with an
  online false-positive rate per group of gateways (device profile when known, otherwise
  all gateways; 20 % quantile of alerts / expected false alerts) that also scales the
  false-alert term, and a two-sided Page–Hinkley test per gateway that cuts its alert window
  to the last 5 s on a change; recall is not adapted;
- **true attack flows** (`ps-anneal-true-a`): the formula with the true `a(t)` — the limit of
  any estimate of `a`;
- **oracle values**: the exact value of every option from ground truth — the limit of any
  value estimate.

## 5. Results on the earlier conditions

### Python simulator, seeds 2–6

| Condition | formula | formula, best hand setting | adaptive | learned, w 1e-4 (unseen) | learned, w 1e-3 (unseen) | DFL (unseen) | true attack flows | oracle values |
|---|---|---|---|---|---|---|---|---|
| x1 | **0.086** | 0.086 | **0.086** | 0.092 | — | 0.096 | 0.094 | 0.106 |
| x3 | 0.255 | 0.255 | 0.256 | 0.250 | **0.248** | **0.248** | 0.245 | 0.230 |
| x5 | 0.379 | 0.344 | 0.368 | 0.352 | **0.343** | 0.345 | 0.358 | 0.327 |
| separation error 5 % | 0.350 | 0.338 | 0.355 | 0.330 | 0.328 | **0.327** | 0.330 | 0.306 |
| drift | 0.340 | 0.325 | 0.334 | 0.322 | 0.316 | **0.314** | 0.337 | 0.291 |

### iFogSim, 20 seeds (x3 and drift)

| Policy | x3 | drift |
|---|---|---|
| formula | 0.254 ± 0.002 | 0.341 ± 0.003 |
| formula, weight 1e-2 | 0.254 ± 0.003 | 0.325 ± 0.002 |
| learned, w 1e-3 (unseen) | 0.247 ± 0.002 | 0.315 ± 0.002 |
| DFL (unseen) | **0.245 ± 0.002** | **0.312 ± 0.002** |
| oracle values | 0.228 ± 0.002 | 0.290 ± 0.002 |

Paired: learned against the hand-tuned formula −0.7 ± 0.2 (x3) and −1.0 ± 0.2 (drift);
DFL against learned −0.2 ± 0.1 and −0.3 ± 0.2. With the measured decision time charged
(x3, 10 seeds, 10 ms annealing budget): formula 0.269 (17 ms per epoch), learned 0.265
(20 ms), DFL 0.262 (20 ms). For reference (seeds 2–21): the static split tuned offline with
labels (`sa-toca`) reaches 0.251 at x3; the split tuned on drift itself 0.329.

## 6. Results on the scenario families

### Python simulator, seeds 2–6

| Family | formula | formula, w 1e-2 | adaptive | learned (unseen) | DFL (unseen) | learned (all) | DFL (all) | true attack flows | oracle values | headroom |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 detectors | 0.349 | 0.395 | **0.279** | 0.288 | 0.333 | 0.264 | 0.262 | 0.256 | 0.238 | 11.0 |
| F2 botnet | **0.356** | 0.358 | 0.367 | 0.366 | 0.360 | 0.361 | 0.358 | 0.357 | 0.315 | 4.1 |
| F3 evasive | **0.318** | 0.533 | 0.409 | 0.335 | 0.335 | 0.322 | 0.322 | 0.315 | 0.262 | 5.6 |
| F4 surges | 0.379 | **0.327** | 0.349 | 0.377 | 0.373 | 0.371 | 0.374 | 0.352 | 0.341 | 3.7 |
| F5 infrastructure | 0.160 | 0.166 | 0.159 | 0.160 | **0.154** | 0.157 | 0.155 | 0.157 | 0.153 | 0.7 |

(headroom: formula minus oracle values, points; *all*: the family was in training)

### iFogSim, 20 seeds (F1, F4, F3)

Seeds 2–6 and 27–41; [`scripts/night_runs7.sh`](../scripts/night_runs7.sh).

| Policy | F1 detectors | F4 surges | F3 evasive | mean |
|---|---|---|---|---|
| formula | 0.356 ± 0.017 | 0.386 ± 0.012 | **0.316 ± 0.004** | 0.353 |
| formula, weight 1e-2 | 0.404 ± 0.019 | **0.329 ± 0.010** | 0.531 ± 0.005 | 0.421 |
| adaptive | **0.279 ± 0.006** | 0.361 ± 0.011 | 0.409 ± 0.007 | 0.350 |
| learned (unseen) | 0.287 ± 0.005 | 0.386 ± 0.013 | 0.334 ± 0.006 | **0.336** |
| DFL (unseen) | 0.345 ± 0.019 | 0.388 ± 0.012 | 0.335 ± 0.007 | 0.356 |
| oracle values | 0.236 ± 0.002 | 0.341 ± 0.012 | 0.260 ± 0.004 | 0.279 |

Paired, learned against: adaptive +0.8 ± 0.4 (F1), +2.6 ± 0.9 (F4), −7.5 ± 0.7 (F3); the
best non-learned policy of each family +0.8 (F1, adaptive), +5.7 ± 0.7 (F4, weight 1e-2),
+1.8 ± 0.4 (F3, formula). On seeds 2–6 the iFogSim means agree with the Python simulator
within 0.013 for every family and policy.

## 7. Over all ten conditions (Python, unseen models)

| Estimate | mean objective | largest shortfall against the best non-oracle estimate |
|---|---|---|
| formula | 0.297 | 7.0 points (F1) |
| formula, weight 1e-2 | 0.313 | 21.5 points (F3) |
| adaptive | 0.296 | 9.1 points (F3) |
| **learned** | **0.285** | **5.0 points** (F4) |
| DFL | 0.289 | 5.4 points (F1) |
| oracle values | 0.257 | — |

## 8. What the results say

1. **The learned estimate is the most robust value estimate.** On conditions it was not
   trained on it has the lowest mean objective and the smallest worst case. It is not the
   best everywhere: every hand-made estimate wins somewhere and fails badly somewhere else
   (the formula on F1, the adaptive formula on F3, the hand-tuned weight on F3).
2. **Where the gain comes from.** On the earlier conditions the learned estimate is as good
   as the formula with the *true* attack flows: estimating `a` better cannot gain more. On
   F1 it uses the device profile, information the formula does not have, and closes 55 %
   (unseen) to 85 % (seen) of an 11-point headroom; online statistics with the same
   information (adaptive) do as well.
3. **What no estimate recovers.** 2–7 points to the oracle values remain everywhere. The
   oracle knows which residual tasks are benign and raises a third of the false alerts or
   less; on F2 and F3 even the true attack flows do not help (what is missing is *which*
   tasks are attacked). On F3 an attacker below the alert noise is invisible to alerts.
4. **Accurate values are not always the best coefficients.** At x1 and on F4 hand-made
   values beat the exact per-task values: a value that is correct task by task ignores how
   the decisions interact over epochs (exploration, false-alert budget).
5. **Decision-focused training** helps on conditions like those it was trained on (iFogSim,
   20 seeds: −0.2 to −0.3 points against the two-stage model; seen families: at least as
   good) and transfers worse to a new kind of condition (F1 unseen: 0.345 against 0.287).
   It learns the trade-offs of its training conditions, and a new kind of detector error
   changes them.
6. **Cost.** About 3 ms per epoch for the prediction; the gain survives charged decision
   time.

## 9. Hypotheses of the plan

| | Hypothesis | Outcome |
|---|---|---|
| H1 | headroom ≥ 3 points in F2, F3, F4 | **confirmed** (4.1, 5.6, 3.7; F1 11.0) |
| H2 | adaptive formula closes ≥ half of the headroom in F1, < a third in F2 and F4 | **confirmed for F1 and F2**; in F4 it closes more |
| H3 | two-stage learning beats the adaptive formula by ≥ 1 point in F2 and F4 (unseen) | **not confirmed** (F2 −0.8 ± 0.5, F4 +2.6 ± 0.9 in iFogSim) |
| H4 | DFL at least as good as two-stage in most families | **mixed**: yes on seen conditions, no on unseen F1 |
| H5 | DFL beats the oracle values somewhere | **not confirmed** (hand-made values do, at x1 and F4) |
| H6 | F3 far from the oracle for every estimate | **confirmed** (≥ 5 points) |

## 10. Limitations

- One annealer and one value form; learning replaces only the attack-flow estimate and two
  weights. A value model per option (not per task) was not tried.
- DFL trains on recorded epochs of `ps-anneal-alerts`; the states it meets when it decides
  itself differ (no retraining on its own runs).
- F3 uses a fixed migration schedule, not an attacker reacting to its own alerts.
- The adaptive formula does not adapt recall (delayed labels are not modelled).
- The families are synthetic; the detector rates are calibrated on CICIoT2023, the device
  profiles and campaign timings are assumptions.

## 11. Reproduce

    python scripts/collect_values.py --variant <condition> --seeds 7 8 ... 16
    python scripts/train_values.py --loo                                   # set v1, earlier conditions
    python scripts/train_values.py --variants f1-detectors ... --features v2 --loo --prefix value-gbm-v2
    python scripts/train_dfl.py [--exclude <condition>]                    # set v1
    python scripts/train_dfl.py --features v2 --variants <families> [--exclude <family>]
    python scripts/robustness.py --variants <...> --policies <...> --seeds 2 3 4 5 6
    python scripts/smokescreen_ifogsim.py --config configs/ids-place-smokescreen-x3.json --variant <...> --policies <...>

Overnight batches: `scripts/learned_eval.sh`, `scripts/hand_tuning.sh`, `scripts/dfl_eval.sh`,
`scripts/night_runs4.sh` … `scripts/night_runs7.sh`. Policies with settings are written
`name@key=value` (for example `ps-anneal-alerts@fa_weight=0.01`).
