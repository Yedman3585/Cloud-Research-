# Scenario families: hand-made, adaptive and learned value estimates

Plan and hypotheses: [learning-plan.md](learning-plan.md) (committed before these runs, with
the amendments listed there). Earlier conditions: [smokescreen.md](smokescreen.md).
Python simulator, seeds 2–6, mean ± 95 % interval over seeds; paired differences in the
text. Runs: [`scripts/night_runs5.sh`](../scripts/night_runs5.sh),
[`scripts/night_runs6.sh`](../scripts/night_runs6.sh).

## The families

Code: [`fogids/families.py`](../src/fogids/families.py). Each family changes one
assumption of the alert formula; everything a family draws comes from the configuration
seed, so every seed is a different instance.

| Family | What changes | Assumption it breaks |
|---|---|---|
| F1 detectors | device profiles per gateway (camera: false-positive rate x3, recall x0.6; hub x1.5 / x0.85; sensor x0.3 / x1); a quarter of the gateways change profile at 30 s | one false-positive rate and recall per model |
| F2 botnet | staged campaigns: scan, brute force, Mirai flood, spreading inside fog subtrees | independent, stationary gateways |
| F3 evasive | low-rate attackers at ~2 alerts per 10 s, moving every 8–12 s (fixed schedule) | attacks show up as excess alerts |
| F4 surges | benign flash crowds x5–x10 on groups of gateways, a third taken for heavy hitters | excess alerts mean attacks |
| F5 infrastructure | other trees, fog capacity x0.5–x2, uplinks x0.25–x1, load x1–x5 | — (load terms stay exact) |

## Competitors

All use the same annealer, budget and repair; only the value estimate differs.

- **formula**: `ps-anneal-alerts`, set once on x3 (false-alert weight 1e-4); **weight 1e-2**: the
  hand-tuned weight that was best on x5, drift and 5 % separation error.
- **adaptive** ([`fogids/adaptive.py`](../src/fogids/adaptive.py)): the formula with an online
  false-positive rate per group of gateways (device profile when known; low quantile of
  alerts / expected false alerts) and a Page–Hinkley change test per gateway that cuts its
  window (L1 + L2 of the plan; recall is not adapted).
- **learned**: gradient-boosted attack-flow estimate with feature set v2 (adds bytes per
  packet, device profile, fog-subtree alerts), false-alert weight 1e-3; **DFL**: the same with
  task-dependent value weights trained on decision regret (feature set v2 in the weights).
  *unseen*: trained on the four other families; *all*: trained on all five (seeds 7–16).
- **true attack flows**: the formula with the true flows of each task; **oracle values**:
  exact per-task values from ground truth.

## Results on the families

| Family | greedy-v2 | formula | formula, weight 1e-2 | adaptive | adaptive, weight 1e-3 | learned (unseen) | DFL (unseen) | learned (all) | DFL (all) | true attack flows | oracle values |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 detectors | 0.399 ± 0.003 | 0.349 ± 0.040 | 0.395 ± 0.056 | 0.279 ± 0.013 | 0.286 ± 0.016 | 0.288 ± 0.014 | 0.333 ± 0.036 | 0.264 ± 0.015 | 0.262 ± 0.011 | 0.256 ± 0.011 | 0.238 ± 0.006 |
| F2 botnet | 0.442 ± 0.032 | 0.356 ± 0.029 | 0.358 ± 0.024 | 0.367 ± 0.027 | 0.375 ± 0.023 | 0.366 ± 0.024 | 0.360 ± 0.022 | 0.361 ± 0.027 | 0.358 ± 0.027 | 0.357 ± 0.029 | 0.315 ± 0.031 |
| F3 evasive | 0.465 ± 0.055 | 0.318 ± 0.017 | 0.533 ± 0.015 | 0.409 ± 0.016 | 0.453 ± 0.014 | 0.335 ± 0.031 | 0.335 ± 0.023 | 0.322 ± 0.020 | 0.322 ± 0.024 | 0.315 ± 0.014 | 0.262 ± 0.011 |
| F4 surges | 0.512 ± 0.021 | 0.379 ± 0.018 | 0.327 ± 0.017 | 0.349 ± 0.019 | 0.346 ± 0.021 | 0.377 ± 0.023 | 0.373 ± 0.019 | 0.371 ± 0.020 | 0.374 ± 0.020 | 0.352 ± 0.014 | 0.341 ± 0.010 |
| F5 infrastructure | 0.269 ± 0.179 | 0.160 ± 0.170 | 0.166 ± 0.150 | 0.159 ± 0.163 | 0.161 ± 0.165 | 0.160 ± 0.172 | 0.154 ± 0.175 | 0.157 ± 0.152 | 0.155 ± 0.156 | 0.157 ± 0.160 | 0.153 ± 0.132 |

## The same competitors on the earlier conditions

Learned and DFL models here are those of [smokescreen.md](smokescreen.md) (feature set v1,
trained without the evaluated condition).

| Condition | formula | adaptive | adaptive, weight 1e-3 | learned (unseen, weight 1e-3) | DFL (unseen) | oracle values |
|---|---|---|---|---|---|---|
| x1 | 0.086 ± 0.008 | 0.086 ± 0.007 | 0.087 ± 0.007 | 0.092 ± 0.009 (weight 1e-4) | 0.096 ± 0.008 | 0.106 ± 0.005 |
| x3 | 0.255 ± 0.008 | 0.256 ± 0.006 | 0.258 ± 0.007 | 0.248 ± 0.007 | 0.248 ± 0.008 | 0.230 ± 0.006 |
| x5 | 0.379 ± 0.007 | 0.368 ± 0.012 | 0.371 ± 0.003 | 0.343 ± 0.007 | 0.345 ± 0.006 | 0.327 ± 0.006 |
| separation error 5 % | 0.350 ± 0.010 | 0.355 ± 0.013 | 0.360 ± 0.011 | 0.328 ± 0.010 | 0.327 ± 0.009 | 0.306 ± 0.009 |
| drift | 0.340 ± 0.005 | 0.334 ± 0.004 | 0.339 ± 0.002 | 0.316 ± 0.004 | 0.314 ± 0.005 | 0.291 ± 0.005 |

## Over all ten conditions

Learned and DFL trained without the evaluated condition (x1: weight 1e-4).

| Estimate | mean objective | largest shortfall against the best non-oracle estimate of a condition |
|---|---|---|
| formula | 0.297 | 7.0 points (F1) |
| formula, weight 1e-2 | 0.313 | 21.5 points (F3) |
| adaptive | 0.296 | 9.1 points (F3) |
| **learned** | **0.285** | **5.0 points** (F4) |
| DFL | 0.289 | 5.4 points (F1) |
| oracle values | 0.257 | — |

## Findings

1. **No estimate is best everywhere; the learned estimate is the most robust.** It has the
   lowest mean and the smallest worst case, on conditions it was not trained on. Every
   hand-made estimate fails somewhere: the formula on F1 (+7 points), the hand-tuned weight
   on F3 (0.533 against 0.318), the adaptive formula on F3 (0.409).
2. **F1, detector quality varies by device:** the adaptive formula (0.279) and the learned
   estimate (0.288 unseen, 0.264 trained on F1) both use the device profile and close
   55–85 % of the 11-point headroom. Online statistics are enough here (H2).
3. **F4, benign surges:** a higher false-alert weight is best (0.327, below the oracle
   values 0.341: exact per-task values are again not the best decision coefficients); the
   adaptive formula gets 3 points, the learned estimates about nothing.
4. **F2 and F3 stay 4–6 points from the oracle values for every estimate.** The formula with
   the true attack flows is no better than the plain formula: the missing piece is
   information about which tasks are attacked, not the size of the attack. For F3 this is
   H6 (an attacker below the alert noise cannot be found from alerts).
5. **F5:** headroom under 1 point; changing the infrastructure does not break the value
   coefficients (the load terms are exact).
6. **DFL against two-stage learning:** with the evaluated family in training, DFL is at
   least as good (F1 0.262 / 0.264, F2 0.358 / 0.361, F5 0.155 / 0.157); without it, DFL
   transfers worse (F1 0.333 against 0.288). Decision-focused weights fit the decision
   trade-offs of the training conditions, which do not carry over to a new kind of
   detector error.

## Hypotheses of the plan

| | Hypothesis | Outcome |
|---|---|---|
| H1 | headroom ≥ 3 points in F2, F3, F4 | **confirmed** (4.1, 5.6, 3.7; F1 11.0) |
| H2 | adaptive formula closes ≥ half of the headroom in F1, < a third in F2 and F4 | **confirmed for F1 and F2**; F4: it closes more than a third (3 of 3.7 points) |
| H3 | two-stage learning beats the adaptive formula by ≥ 1 point in F2 and F4 (unseen family) | **not confirmed**: F2 −0.8 ± 0.5, F4 +3.1 ± 2.6 |
| H4 | DFL at least as good as two-stage in ≥ 4 of 6 families | **mixed**: on families seen in training yes; on unseen families no (F1) |
| H5 | DFL beats the oracle values somewhere | **not confirmed** (the hand-tuned weight does, in F4) |
| H6 | F3 far from the oracle for every estimate | **confirmed** (≥ 5.3 points) |

## Next

- iFogSim, 20 seeds, for F1, F3 and F4 (where the estimates differ most).
- Paper: method (exact processor-sharing epoch model and annealer), alert-based values,
  learned and decision-focused values, and these families as the test of when learning
  pays off.
