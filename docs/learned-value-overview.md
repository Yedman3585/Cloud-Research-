# Learned Value Estimation for the QUBO Scheduler

## Context: what the scheduler decides and why values matter

Every 100 ms epoch, the scheduler places IDS analysis tasks across gateway, fog and cloud nodes. It formulates this as a QUBO problem: each task has several options (node, detector model, sampling fraction, extraction site), and the annealer picks the combination that maximises total value minus contention cost within per-node deadline constraints.

The **load terms** (processor-sharing finish times, link queues, budgets) are computed exactly from the instance. The **value of each option** — how much analysing a particular task reduces missed detections — is not known at decision time because the scheduler does not know which tasks contain attacks.

Previously, the value was set by a fixed formula that converts the gateway's alert statistics over a 40-second window into an estimate of attack flows per task. This works well on the reference scenario (load ×3), but the formula's parameters are tuned once and do not adapt to conditions it was not designed for.

## What we built

A **gradient-boosted trees regression model** (`HistGradientBoostingRegressor`, Poisson loss) that predicts the number of attack flows in each task from features the scheduler can see at decision time. The prediction replaces the fixed formula inside the same QUBO; everything else (load model, annealer, budget, repair) is unchanged.

### Features (all observable at decision time)

| Feature | Description |
|---|---|
| `flood` | 1 if the task belongs to the flood stream, 0 otherwise |
| `prefilter_score` | fraction of packets flagged by the lightweight header filter |
| `n_flows` | number of network flows in the task |
| `packets` | total packets |
| `criticality` | gateway criticality weight |
| `deadline_s` | task deadline in seconds |
| `alerts_{10,20,40}` | cumulative IDS alerts at this gateway over 10, 20 and 40 s |
| `expected_false_{10,20,40}` | expected false alerts (from analysed benign traffic) |
| `analysed_recall_{10,20,40}` | fraction of traffic actually analysed at the gateway |
| `share_{10,20,40}` | gateway's share of total alerts across the tree |
| `z_{10,20,40}` | standardised alert excess (alerts − expected) / std |

Total: 21 features.

### Training target

The **true attack flows** the value formula represents: low-rate attack flows in a residual task, all attack flows in a flood task. This is the ground-truth number that the fixed formula tries to approximate from alerts.

### Training data

We run `ps-anneal-alerts` (the existing method) on training seeds and record, for every task, its features and the true attack flows. The data comes from:

- Seeds 7–26 on the reference scenario (load ×3)
- Seeds 7–16 on four other conditions (load ×1, load ×5, separation error 5%, drift)
- Total: 538,000 tasks

Evaluation seeds (2–6) are disjoint from training.

### Model

`HistGradientBoostingRegressor` from scikit-learn:

- Loss: Poisson (count data, non-negative)
- 400 boosting iterations, learning rate 0.08
- Max 31 leaf nodes, min 40 samples per leaf
- Inference: ~3 ms per epoch (well within the 100 ms budget)

### Training scheme

**Two-stage** (predict-then-optimise): the model is trained to minimise Poisson loss on the true attack flows, independently of how well those predictions lead to good scheduling decisions. The predicted values are then plugged into the QUBO coefficients and the annealer solves as usual.

**Leave-one-out** (`-loo`): a separate model is trained for each condition, excluding that condition from the training data, to test generalisation to unseen scenarios.

## Results

### Prediction quality

Spearman rank correlation with true low-rate flows (residual tasks, held-out seeds):

- Learned model: 0.79–0.84
- Fixed formula: 0.43–0.73
- On an unseen condition: 0.46–0.75

### Scheduling quality (Python simulator, seeds 2–6)

| Condition | `ps-anneal-alerts` (formula) | `-learned-loo` | `-learned` | `-true-a` | Oracle values |
|---|---|---|---|---|---|
| load ×1 | **0.086** | 0.092 (+0.6) | 0.095 (+0.9) | 0.094 | 0.106 |
| ×3 (reference) | 0.255 | 0.250 (−0.5) | 0.249 (−0.7) | 0.245 | 0.230 |
| load ×5 | 0.379 | **0.352 (−2.7)** | 0.355 (−2.4) | 0.358 | 0.327 |
| sep. error 5% | 0.350 | 0.330 (−2.0) | **0.329 (−2.0)** | 0.330 | 0.306 |
| drift | 0.340 | **0.322 (−1.8)** | 0.330 (−1.0) | 0.337 | 0.291 |

Lower is better. Differences are paired by seed; ±95% confidence intervals range 0.3–1.0 pp.

### Hand-tuned comparison (best per-condition settings, favouring the formula)

| Condition | Formula (default) | Formula (best setting) | Learned (one setting) | Oracle |
|---|---|---|---|---|
| ×3 | 0.257 | 0.255 | **0.248** | 0.230 |
| ×5 | 0.379 | 0.344 | **0.343** | 0.327 |
| sep. error 5% | 0.350 | 0.338 | **0.328** | 0.306 |
| drift | 0.341 | 0.325 | **0.316** | 0.291 |

The learned model with one fixed setting (false-alert weight 1e-3, floor 3) beats the formula tuned per-condition on 3 of 4 scenarios; at ×3 it also beats SA-TOCA (0.248 vs 0.252).

## Key findings

1. **The learned estimate helps on every hard condition**, gaining 1–2.7 points, including on conditions not in its training data (leave-one-out). It closes 21–51% of the gap between the formula and the oracle.

2. **It matches the formula with true attack flows** (`-true-a`): predicting the flows more accurately cannot gain more. The remaining 1.6–2.5 points to the oracle are in the *form* of the value (the oracle raises 3–4× fewer false alerts), not in the flow estimate.

3. **Better prediction ≠ better decisions**: on drift, the model trained on all conditions predicts better (Spearman 0.79 vs 0.66) but decides worse (0.330 vs 0.322). This motivates decision-focused learning.

4. **Self-fulfilling features**: without the floor of 3 attack flows, the model stops analysing tasks with no alerts, which then never generate alerts (objective degrades to 0.71). The floor acts as forced exploration.

## What this means for the paper

This is the first empirical evidence in the project that learning the QUBO coefficients improves scheduling decisions compared to a hand-designed formula, and that the improvement transfers to unseen conditions. The finding that the prediction ceiling (`-true-a`) has been reached motivates the next step: **decision-focused learning** (DFL), which trains the model on the quality of the scheduling decisions themselves (via SPO+ or Fenchel–Young loss) rather than on prediction accuracy, targeting the remaining 1.6–2.5 points to the oracle.

## Code

| File | Role |
|---|---|
| `scripts/collect_values.py` | Records training data (features + ground truth) from simulator runs |
| `scripts/train_values.py` | Trains the GBT model (Poisson loss), reports held-out metrics |
| `src/fogids/learned.py` | `LearnedPSAnneal` policy: loads the model, predicts at each epoch |
| `scripts/learned_eval.sh` | Evaluation script across conditions |
| `scripts/hand_tuning.sh` | Grid search over formula and learned hyperparameters |
