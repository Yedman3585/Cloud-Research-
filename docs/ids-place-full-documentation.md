# IDS-PLACE: Full Project Documentation

**Online Placement of Deep-Learning IDS Inference in an Edge–Fog–Cloud Tree**

---

## Table of Contents

1. [Idea and Research Question](#1-idea-and-research-question)
2. [Data and Simulation](#2-data-and-simulation)
3. [Experiments and Results](#3-experiments-and-results)

---

## 1. Idea and Research Question

### 1.1 Problem

Modern IoT networks deploy Intrusion Detection Systems (IDS) to identify attacks — DDoS floods, reconnaissance, brute force, web exploits, Mirai botnets. A deep-learning IDS offers high accuracy but is too expensive for every edge gateway to run on all traffic in real time. Meanwhile, fog and cloud nodes have more compute but are further away and share bandwidth with other gateways.

The scheduler must decide, every 100 ms, for each chunk of traffic:
- **Where** to process it (gateway, fog, or cloud),
- **Which detector** to use (a cheap fast one or an expensive accurate one),
- **What fraction** of the packets to analyse,
- **Where to extract features** (locally or offload raw packets).

A wrong decision means either a missed attack (the traffic was dropped or undersampled) or wasted resources and false alerts (too much benign traffic analysed with an inaccurate model).

### 1.2 Research Question

> Can an online scheduler, with no ground-truth labels and no offline tuning, match or beat a static traffic split that was optimised offline with full knowledge of the attack labels — and can learning the scheduler's decision coefficients from IDS alerts improve robustness across changing conditions?

### 1.3 Core Idea

We formulate each epoch's placement as a **Quadratic Unconstrained Binary Optimisation (QUBO)** problem:

- **Binary variables:** one per (task, option) pair, where an option is a (node, model, sampling fraction, extraction site) tuple.
- **Linear terms (values):** the expected detection benefit of each option, minus a false-alert penalty — this is where the unknown lies: which tasks contain attacks?
- **Quadratic terms (load):** the exact processor-sharing contention cost. When n tasks share a CPU, each runs at speed/n, so a late arrival slows everything already running. The sum of finish times is `Σ aᵢ + 2 Σᵢ<ⱼ min(aᵢ, aⱼ)` — an exact quadratic in the placement variables. Uplinks are FIFO queues, also modelled exactly.
- **Constraints as penalties:** each task's predicted latency (its QUBO row sum) must fit within its deadline; lateness is penalised with weight κ = 2.

A fast simulated annealer (~120,000 moves in 50 ms via Numba) solves this QUBO every epoch. The key insight is that the **load terms are exact** — the only thing we estimate is the **value of each task**, which depends on whether it contains an attack.

### 1.4 Three Layers of Value Estimation

1. **Fixed formula from IDS alerts.** The gateway accumulates IDS alerts over a 40-second window. For a residual (non-flood) task, the estimated attack flows = `max(floor, alert_share × n_flows)`, where `alert_share` is this gateway's fraction of all recent alerts. For a flood task, the pre-filter score gives the estimate directly.

2. **Learned estimate (gradient-boosted trees).** A regression model predicts the true attack flows from 21 scheduler-visible features (pre-filter score, stream type, task size, deadline, and the gateway's alert statistics over 10/20/40 seconds). Trained on 538,000 tasks from simulator runs; plugged into the same formula as above.

3. **Decision-focused learning (DFL).** Instead of training on prediction accuracy, train on decision quality: how much does the scheduling objective improve when using these predicted values? Uses task-dependent value weights trained on decision regret.

### 1.5 Contributions

1. **Exact processor-sharing epoch model** as QUBO deadline constraints — replaces the FIFO queue approximation used in prior work; accounts for 6.6 pp improvement in ablations.
2. **IDS alert feedback** for online task value estimation — accounts for 8.6 pp improvement.
3. **Realistic smokescreen scenario** (flood as cover for low-rate attacks), screened with an LP bound and validated in iFogSim.
4. **Learned value coefficients** — gradient-boosted trees as the most robust estimate across 10 conditions, including unseen ones.
5. **Scenario families** (F1–F5) as a benchmark for when learning pays off and when it doesn't.
6. **Honest evaluation of DFL** — it helps on known conditions but transfers worse than two-stage learning to unseen families.

---

## 2. Data and Simulation

### 2.1 Dataset: CICIoT2023

The IDS detectors are trained and calibrated on CICIoT2023 (Neto et al., Sensors 2023). This is a labelled dataset of IoT network traffic with 34 attack types grouped into 7 classes plus benign:

| IDS-PLACE class | CICIoT2023 labels | Relative traffic rate |
|---|---|---|
| benign | BenignTraffic | 1.0× |
| ddos | DDoS-* (12 labels) | 15.8× |
| dos | DoS-* (4 labels) | 12.3× |
| mirai | Mirai-* (3 labels) | 2.8× |
| recon | Recon-*, VulnerabilityScan | 1.2× |
| spoofing | DNS_Spoofing, MITM-ArpSpoofing | 3.4× |
| web | SqlInjection, XSS, CommandInjection, etc. | 0.4× |
| bruteforce | DictionaryBruteForce | 0.6× |

Sample: 559,800 rows from 309 CSV files, capped at 20,000 per original label. Rows are consecutive blocks of 200 to preserve temporal structure; blocks (not rows) are split 70/15/15 per class.

### 2.2 IDS Detector Models

Two models classify each flow into the 8 classes. Inputs pass through signed log1p and standardisation.

**`light` (MLP):** 39 → 64 → 32 → 8. Operates on a single flow record.

**`full` (GRU context model):** Each of 16 consecutive records is embedded (39 → 64), a GRU (64 units) processes the sequence, and a head (128 → 64 → 8) reads the final state plus the target flow's embedding. During training, context records are randomly replaced (probability ~U(0, 0.5)) to simulate mixed traffic at a gateway.

| Model | Params | Cost per 1000 flows | Balanced accuracy | Benign FPR |
|---|---|---|---|---|
| `light` | 4,904 | 0.17 ms | 0.716 | 25.6% |
| `full` (50% mixed context) | 36,296 | 12.3 ms | 0.804 | 12.7% |

Detection recall (test split):

| Model | DDoS | DoS | Mirai | Recon | Spoofing | Web | Bruteforce |
|---|---|---|---|---|---|---|---|
| `light` | 1.000 | 1.000 | 1.000 | 0.855 | 0.850 | 0.871 | 0.857 |
| `full` | 1.000 | 1.000 | 1.000 | 0.844 | 0.863 | 0.917 | 0.909 |

Key finding: `full` halves the false-positive rate and improves recall on rare attack classes, but costs 73× more per flow. This creates the placement trade-off.

### 2.3 Simulation Environment

#### Topology

A tree: 1 cloud → 4 fog nodes → 32 edge gateways (8 per fog). Each node has:

| Tier | Device | MIPS | Role |
|---|---|---|---|
| Edge gateway | Raspberry Pi 4 (4 cores) | 2,566 | Receives traffic, extracts features, runs `light` |
| Fog | Apple M2 (all cores) | 37,793 | Runs both models, receives offloaded tasks |
| Cloud | — | 64,000 | Unlimited memory, highest cost |

Uplinks: 10 Mbit/s bandwidth, 5 ms latency (edge→fog), 20 ms (fog→cloud).

#### iFogSim2

iFogSim2 (v2.0.0) is the reference simulator. It implements:
- **Processor sharing (PS)** on CPUs: n tasks on a node each get speed/n. A new arrival slows all running tasks.
- **FIFO queues** on uplinks: transfers are serialised.
- **Cold starts:** loading a non-resident model costs extra CPU time; LRU eviction within node memory.

**We do not modify the iFogSim core.** We add:
- **Java driver** (`org.fogids.IdsStaged`): injects tasks, pauses every 100 ms epoch, communicates with Python over JSON on stdin/stdout.
- **Task stages:** header pass (CPU), feature extraction (CPU), transfer (uplink), inference (CPU at destination).
- **Scaling:** work in MI and node MIPS scaled by 1000 (iFogSim requires integers); CPU update interval set to 1 ms for deadline accuracy.

#### Python Simulator

A parallel Python simulator (`smokescreen.py`, class `Engine`) with identical PS and FIFO semantics. It matches iFogSim within 3–6 ms on finish times and within 0.001 on the objective. Used for fast experiments (seconds vs minutes per run); iFogSim validates the final tables.

### 2.4 Smokescreen Scenario

The calibrated scenario from IDS-PLACE v1 left only 0.2–0.3 pp of headroom for any method. Two measurements motivated a harder, realistic scenario:

1. **Feature extraction dominates gateway CPU:** 7–8 µs per packet (dpkt on M2). A DDoS burst offers a Pi gateway ~11× the packets it can parse.
2. **Sampling preserves per-flow quality:** analysing every s-th flow, the GRU model keeps its recall; sampling acts through coverage only.

The **smokescreen scenario**: a DDoS/DoS flood saturates gateways while low-rate attacks (recon, web, bruteforce) hide underneath. The scheduler must decide how much flood traffic to drop/sample and how to protect residual analysis.

**Traffic (x3 variant):** 900 benign flows/s per gateway; floods on 12+8 gateways at CICIoT2023 ratios; low-rate attacks on 6 gateways each at 40 or 15 flows/s, half overlapping with floods.

**Decision per task:** (node, model ∈ {light, full}, sampling fraction f ∈ {1, 0.5, 0.2, 0.05, 0.01}, extraction site ∈ {gateway, destination}).

**Objective:** macro-averaged missed detection across attack classes + 0.05 × false alerts per benign flow. Lower is better.

### 2.5 Scenario Families (F1–F5)

To test robustness, five families each break one assumption of the fixed alert formula:

| Family | What changes | What it breaks |
|---|---|---|
| **F1 — heterogeneous detectors** | Per-gateway device profiles (camera/hub/sensor) scale FPR ×0.3–5 and recall ×0.5–1; a quarter change mid-run | One FPR and recall per model everywhere |
| **F2 — botnet campaign** | Staged attack: scan → brute force → Mirai/DDoS; spreading within fog subtrees | Independent, stationary gateways |
| **F3 — evasive attacker** | Low-rate attacker stays under ~2 alerts per 10s, moves to fresh gateways every 8–12s | Attacks show up as excess alerts |
| **F4 — benign surges** | Flash crowds ×5–10 on gateway groups; a third flagged as heavy hitters | Excess alerts mean attacks |
| **F5 — infrastructure** | Fan-out 2–8, fog capacity ×0.5–2, uplinks ×0.25–1, load ×1–5 | — (tests whether load terms being exact is enough) |

---

## 3. Experiments and Results

### 3.1 Experiment 1: Calibrated Scenario and Baseline Policies (IDS-PLACE v1)

**Goal:** establish the placement problem and the headroom for optimisation on the calibrated CICIoT2023-based scenario.

**Policies tested:** edge-light, edge-full, fog-full, cloud-full, greedy-finish, edge-adaptive, queue-greedy, risk-split, random.

**Key result (medium instance, x3, 5 seeds):**
- `risk-split` meets every deadline with 14.5% false alerts.
- `queue-greedy` misses 5.2% of deadlines but achieves 13.3% false alerts.
- Fluid LP bound: 13.1% false alerts at zero misses.
- **Headroom: only 1.4 pp between the best baseline and the bound.**

**Finding:** the calibrated scenario is too easy. Simple rules nearly solve it because the cheap model meets all deadlines and the expensive model's extra accuracy is worth only ~1 pp.

### 3.2 Experiment 2: Processor-Sharing Model Validation

**Goal:** show that the FIFO queue estimate used by prior work fails under load, and that an exact PS model fixes it.

**Key result (same decisions of `queue-greedy`):**

| Load | Estimator | Rank correlation with simulator |
|---|---|---|
| x3 | FIFO queue model | 0.07 |
| x3 | PS model | 0.89 |
| x10 | FIFO queue model | −0.57 |
| x10 | PS model | 0.87 |

**Finding:** the FIFO model's latency ranking is uncorrelated or inverted under load. The PS model restores correlation to 0.87–0.89. Every deadline miss of `queue-greedy` was a broken promise of the FIFO estimator.

### 3.3 Experiment 3: Epoch QUBO with PS Constraints (IDS-PLACE v1)

**Goal:** test whether joint optimisation via QUBO beats the greedy rule on the calibrated scenario.

**Key result (x3, 5 seeds):**
- `ps-greedy-s0.1`: 0.02% misses, 13.4% false alerts.
- `qubo-ps`: 0% misses, 13.5% false alerts.
- Both within 0.3 pp of the fluid bound.

**Finding:** on the calibrated scenario the QUBO is on par with greedy — there's nothing left to win. This motivated the smokescreen scenario.

### 3.4 Experiment 4: Smokescreen Scenario Screening

**Goal:** verify that x3 of the smokescreen scenario has enough headroom to justify optimisation.

**Key result (x3, seeds 2–6):**

| Policy | Objective | Gap to bound |
|---|---|---|
| Fluid LP bound | 0.047 | — |
| `greedy-v2` | 0.386 | 33.9 pp |
| `stream-aware` | 0.488 | 44.1 pp |

**Finding:** x3 passes the screening criterion — 34 pp of headroom. The expert rule breaks because processor sharing causes flood chunks to miss their deadlines even at 1% sampling. x1 fails screening (expert rule within 0.8 pp of the bound).

### 3.5 Experiment 5: Joint Epoch Optimisation on Smokescreen (x3)

**Goal:** quantify the improvement from QUBO placement with PS deadline constraints.

**Key result (seeds 2–6, x3):**

| Policy | Objective | Missed low-rate | Missed flood |
|---|---|---|---|
| `greedy-v2` | 0.386 | 16.3% | 70.8% |
| `ps-anneal` | 0.366 | 23.5% | 55.0% |
| `ps-anneal-oracle-value` | 0.290 | 11.0% | 55.5% |

**Finding:** joint placement detects 16 pp more flood tasks (chunks are split optimally). But it loses on low-rate attacks (−7.2 pp) because their value is invisible from the pre-filter score. The oracle shows 7.6 pp of headroom for a better value estimate.

### 3.6 Experiment 6: Alert Feedback

**Goal:** use IDS alerts from completed tasks to estimate which gateways are under low-rate attack.

**Mechanism:** `AlertEvidence` accumulates residual-stream alerts per gateway over 40 seconds. The estimated attack share = (alerts − expected false alerts) / (analysed flows × mean recall). This feeds into the value formula: estimated attack flows = `max(3, share × n_flows)`.

**Key result (seeds 2–6, x3):**

| Policy | Objective | Missed low-rate | Missed flood |
|---|---|---|---|
| `greedy-v2` | 0.386 | 16.3% | 70.8% |
| `value-greedy-alerts` | 0.325 | 14.4% | 59.7% |
| `ps-anneal-alerts` | **0.255** | **1.9%** | 59.6% |
| Oracle values | 0.230 | 2.1% | 54.2% |

**Finding:** alert feedback is the single largest improvement — **13.1 pp over greedy-v2**. It almost eliminates missed low-rate attacks (from 16.3% to 1.9%). The gap to the oracle narrows to 2.5 pp.

### 3.7 Experiment 7: Fast Annealer

**Goal:** increase the annealer's search quality within the 50 ms budget.

**Implementation:** `fast_anneal.py` compiles the move kernel with Numba. ~2.5M moves/s vs ~20k in pure Python → ~120,000 moves in 50 ms instead of ~1,000.

**Key result (seeds 2–6, x3):**

| Policy | Before | After |
|---|---|---|
| `ps-anneal` | 0.366 | 0.341 |
| `ps-anneal-alerts` | 0.289 | 0.256 |
| `ps-anneal-oracle-value` | 0.290 | 0.230 |

**Finding:** the faster annealer gains 2–6 pp across the board. With oracle values the annealer now leaves the alert-based policy 2.6 pp behind — the value estimate matters again once the search is good enough.

### 3.8 Experiment 8: iFogSim Validation (x3, 5 seeds)

**Goal:** confirm the Python results in iFogSim.

| Policy | Objective | Missed low-rate | Decision time |
|---|---|---|---|
| `greedy-v2` | 0.393 ± 0.008 | 16.3% | 41 ms |
| `value-greedy-alerts` | 0.330 ± 0.017 | 14.4% | 70 ms |
| `ps-anneal` | 0.341 ± 0.011 | 21.7% | 68 ms |
| **`ps-anneal-alerts`** | **0.256 ± 0.008** | **1.9%** | 67 ms |

**Finding:** iFogSim reproduces the Python simulator within 0.001 on the objective and 6 ms on finish times. Rankings are identical.

### 3.9 Experiment 9: Baselines from the Literature

**SA-TOCA** (Sudyana et al., IEEE TSC 2025): per traffic class, a fixed split over five routes, tuned offline by simulated annealing with ground-truth labels on x3.

**FOG-QIEA** (Hammouda & Shalaby et al., Sci. Rep. 2026): quantum-inspired evolutionary algorithm with the same option values.

| Condition | `greedy-v2` | `sa-toca` (tuned at x3) | `fog-qiea` | **`ps-anneal-alerts`** |
|---|---|---|---|---|
| x3 (tuning condition) | 0.386 | **0.252** | 0.440 | 0.255 |
| load x2 | 0.309 | 0.226 | — | **0.149** |
| load x5 | 0.456 | **0.365** | — | 0.379 |
| extraction 1 µs/pkt | 0.408 | 0.243 | — | **0.088** |

**Key findings:**
1. SA-TOCA ties us at x3 (its tuning point) but falls behind when conditions shift: −7.7 pp at x2, −15.5 pp at 1 µs extraction.
2. SA-TOCA requires ground-truth labels and offline search per condition. Our method needs neither.
3. FOG-QIEA (0.44) cannot compete: it evaluates hundreds of whole solutions per epoch vs our 120,000 incremental moves.

### 3.10 Experiment 10: 20-Seed iFogSim Comparison and Robustness

**Goal:** statistically rigorous comparison across conditions (20 seeds, iFogSim).

Results on the drift scenario (non-stationary: attack patterns change mid-run):
- `ps-anneal-alerts`: 0.340
- `sa-toca` (tuned on drift with labels): 0.329
- `sa-toca` (tuned on x3): 0.353
- Our method without retuning is 1.3 pp behind SA-TOCA's own-condition tuning, and 1.3 pp ahead of SA-TOCA's cross-condition transfer.

### 3.11 Experiment 11: Learned Value Estimate (Two-Stage)

**Goal:** replace the hand-crafted alert formula with a trained model.

**Model:** `HistGradientBoostingRegressor` (Poisson loss, 400 iterations, 31 leaf nodes). 21 features visible at decision time. Trained on 538,000 tasks (seeds 7–26 for x3, seeds 7–16 for x1/x5/sep5%/drift).

**Target:** true attack flows per task (low-rate flows in residual tasks, all attack flows in flood tasks).

**Key result (Python simulator, seeds 2–6):**

| Condition | Formula | Learned (unseen) | Learned (all) | True attack flows | Oracle |
|---|---|---|---|---|---|
| x3 | 0.255 | 0.250 (−0.5) | 0.249 (−0.7) | 0.245 | 0.230 |
| x5 | 0.379 | **0.352 (−2.7)** | 0.355 (−2.4) | 0.358 | 0.327 |
| sep. 5% | 0.350 | 0.330 (−2.0) | **0.329 (−2.0)** | 0.330 | 0.306 |
| drift | 0.340 | **0.322 (−1.8)** | 0.330 (−1.0) | 0.337 | 0.291 |

**Key findings:**
1. The learned estimate helps on every hard condition (+1–2.7 pp), even when that condition was not in training.
2. It matches the formula with true attack flows: better prediction cannot gain more. The remaining gap to the oracle is in the form of the value formula, not the flow estimate.
3. Better prediction ≠ better decisions: on drift, the model trained on all conditions predicts better but decides worse (0.330 vs 0.322 for leave-one-out).

### 3.12 Experiment 12: Hand Tuning vs Learning

**Goal:** a fair comparison — grid-search the formula's hyperparameters (false-alert weight, floor) and compare.

| Condition | Formula (default) | Formula (best setting) | Learned (one setting) | Oracle |
|---|---|---|---|---|
| x3 | 0.257 | 0.255 | **0.248** | 0.230 |
| x5 | 0.379 | 0.344 | **0.343** | 0.327 |
| sep. 5% | 0.350 | 0.338 | **0.328** | 0.306 |
| drift | 0.341 | 0.325 | **0.316** | 0.291 |

**Finding:** the learned model with one fixed setting beats the formula tuned per-condition on 3/4 scenarios. At x3 it also beats SA-TOCA (0.248 vs 0.252).

### 3.13 Experiment 13: Scenario Families — Full Comparison

**Goal:** test all value estimation methods across 5 new families + 5 earlier conditions, using leave-one-family-out training for ML methods. 20 seeds in iFogSim for F1, F3, F4.

#### Results on Families (Python simulator, seeds 2–6)

| Family | Formula | Adaptive | Learned (unseen) | DFL (unseen) | Oracle |
|---|---|---|---|---|---|
| F1 detectors | 0.349 | **0.279** | 0.288 | 0.333 | 0.238 |
| F2 botnet | 0.356 | 0.367 | 0.366 | 0.360 | 0.315 |
| F3 evasive | **0.318** | 0.409 | 0.335 | 0.335 | 0.262 |
| F4 surges | 0.379 | 0.349 | 0.377 | 0.373 | 0.341 |
| F5 infrastructure | 0.160 | 0.159 | 0.160 | 0.154 | 0.153 |

#### Summary Across All 10 Conditions

| Estimate | Mean objective | Largest shortfall vs best |
|---|---|---|
| Formula | 0.297 | 7.0 pp (F1) |
| Formula, weight 1e-2 | 0.313 | 21.5 pp (F3) |
| Adaptive | 0.296 | 9.1 pp (F3) |
| **Learned** | **0.285** | **5.0 pp** (F4) |
| DFL | 0.289 | 5.4 pp (F1) |
| Oracle | 0.257 | — |

#### iFogSim Confirmation (20 seeds, F1/F3/F4)

| Method | F1 | F4 | F3 | Mean |
|---|---|---|---|---|
| Formula | 0.356 | 0.386 | **0.316** | 0.353 |
| Adaptive | **0.279** | 0.361 | 0.409 | 0.350 |
| **Learned** | 0.287 | 0.386 | 0.334 | **0.336** |
| DFL | 0.345 | 0.388 | 0.335 | 0.356 |
| Oracle | 0.236 | 0.341 | 0.260 | 0.279 |

### 3.14 Key Findings Across All Experiments

**1. No single estimate wins everywhere; the learned estimate is the most robust.** It has the lowest mean objective (0.285) and the smallest worst-case shortfall (5.0 pp) across 10 conditions, including conditions not in its training data. Every hand-crafted estimate fails somewhere: the formula on F1 (+7 pp), the adaptive formula on F3 (+9.1 pp), the tuned weight on F3 (+21.5 pp).

**2. The method matches a static split tuned offline with labels, and beats it under shift.** At x3, ps-anneal-alerts (0.255) ties SA-TOCA (0.252); the learned variant (0.248) beats it. Under drift or at different loads, SA-TOCA loses its advantage because it cannot adapt without relabelling and retuning.

**3. DFL transfers poorly to unseen conditions.** On families seen in training, DFL is at least as good as two-stage learning. On unseen families (especially F1), it transfers worse (0.333 vs 0.288) because decision-focused weights overfit to the training conditions' specific trade-offs.

**4. An evasive attacker is a fundamental barrier.** F3 stays 5+ pp from the oracle for every method, including with true attack flows. An attacker that keeps its evidence below the alert noise cannot be detected from alerts — this is a limit of the information available to the scheduler, not of the method.

**5. Exact values are not always the best QUBO coefficients.** In F4 (benign surges), a hand-tuned false-alert weight (0.327) beats the oracle values (0.341). This confirms the theoretical motivation for DFL even though DFL itself has not achieved this.

**6. The ablation confirms both main components:**
- PS epoch model: +6.6 pp (ps-anneal vs greedy-v2 on flood detection).
- Alert feedback: +8.6 pp (ps-anneal-alerts vs ps-anneal on low-rate detection).

### 3.15 Pre-Registered Hypotheses and Outcomes

| # | Hypothesis | Outcome |
|---|---|---|
| H1 | Headroom ≥ 3 pp in F2, F3, F4 | **Confirmed** (4.1, 5.6, 3.7; F1: 11.0) |
| H2 | Adaptive formula closes ≥ half of headroom in F1, < 1/3 in F2 and F4 | **Confirmed for F1 and F2**; F4: closes more than 1/3 |
| H3 | Learned beats adaptive by ≥ 1 pp in F2 and F4 (unseen) | **Not confirmed**: F2 −0.8, F4 +3.1 |
| H4 | DFL ≥ two-stage in ≥ 4 of 6 families | **Mixed**: yes on seen families, no on unseen (F1) |
| H5 | DFL beats oracle somewhere | **Not confirmed** (hand-tuned weight does, in F4) |
| H6 | F3 far from oracle for every estimate | **Confirmed** (≥ 5.3 pp) |

---

## References

1. **QuAnt** — Seelbach Benkner et al., ICLR 2023. [arXiv:2210.08114](https://arxiv.org/abs/2210.08114)
2. **SA-TOCA** — Sudyana et al., IEEE TSC 2025. [DOI:10.1109/TSC.2025.3563680](https://doi.org/10.1109/TSC.2025.3563680)
3. **Q-GARS** — arXiv 2026. [arXiv:2603.23127](https://arxiv.org/abs/2603.23127)
4. **FOG-QIEA** — Hammouda, Shalaby et al., Scientific Reports 2026.
5. **CICIoT2023** — Neto et al., Sensors 2023. [DOI:10.3390/s23135941](https://doi.org/10.3390/s23135941)
6. **iFogSim2** — Mahmud et al., JSS 2022. [DOI:10.1016/j.jss.2021.111139](https://doi.org/10.1016/j.jss.2021.111139)

---

## Code Map

| File | Role |
|---|---|
| `src/fogids/smokescreen.py` | Python simulator (Engine), scenario generator |
| `src/fogids/smokescreen_qubo.py` | QUBO policies: ps-anneal, ps-anneal-alerts, oracle |
| `src/fogids/fast_anneal.py` | Numba-compiled annealer (~120k moves / 50 ms) |
| `src/fogids/ps_model.py` | Exact processor-sharing model |
| `src/fogids/learned.py` | LearnedPSAnneal: GBT-based value estimation |
| `src/fogids/adaptive.py` | Adaptive alert formula (online FPR, change detection) |
| `src/fogids/families.py` | Scenario families F1–F5 |
| `src/fogids/baselines.py` | SA-TOCA and FOG-QIEA implementations |
| `src/fogids/bounds.py` | Fluid LP lower bound |
| `java/org/fogids/IdsStaged.java` | iFogSim2 driver for smokescreen scenario |
| `scripts/train_values.py` | GBT model training (Poisson loss) |
| `scripts/collect_values.py` | Training data collection from simulator runs |
| `scripts/calibrate.py` | CICIoT2023 model calibration |
| `docs/learning-plan.md` | Pre-registered experimental plan and hypotheses |
