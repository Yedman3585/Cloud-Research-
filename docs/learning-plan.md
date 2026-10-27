# Learned value coefficients: scenario families, competitors and hypotheses

This plan is committed **before** the experiments it describes. Families, competitors,
metrics and hypotheses below are fixed now; results are reported against them whether
they confirm the hypotheses or not.

## Question

The epoch model (see [smokescreen.md](smokescreen.md)) has two kinds of coefficients:

- **load terms** (processor-sharing finish times, budgets): computed exactly from the
  instance; nothing to learn;
- **task values** (how much analysing a task with a given model and sampling fraction
  reduces the objective): unknown at decision time, currently estimated by a fixed
  formula from the gateway's alerts over 40 s.

Question: across *different* tasks, does a learned value estimate give better decisions
than the formula, with and without online adaptation of the formula, and does training
it on decision quality (decision-focused learning) beat training it on value accuracy?

Measured headroom on the existing variants (alert estimate minus exact values, seeds 2–6):
−1.9 points at x1, 0.5 at x2, 2.5 at x3, 4.4 with 5 % separation error, 4.9 on drift,
5.2 at x5. At x1 exact per-task values are worse than the formula, so value accuracy and
decision quality already diverge.

## Scenario families

Each family changes one assumption the formula makes. Topology, models, objective and its
weights stay as in `ids-place-smokescreen-x3` unless stated. Everything a family hides
from the scheduler is drawn by the generator; what the scheduler may observe is listed.

| Family | What the generator does | Assumption of the formula it breaks | Scheduler observes |
|---|---|---|---|
| **F0 reference** | current variants: x3, x5, separation 5 %, drift | — | as now |
| **F1 heterogeneous detectors** | each gateway gets a device profile (cameras, sensors, hubs); a model's false-positive rate is scaled ×0.3–×5 and its recall per class ×0.5–×1 by profile; profiles change once mid-run on a quarter of the gateways | one false-positive rate and one recall per model everywhere | device mix of the gateway, alerts |
| **F2 botnet campaign** | staged attack: scan → brute force → Mirai/DDoS; a compromised gateway infects 1–3 gateways of its fog subtree after 5–15 s | gateways are independent and stationary | alerts of neighbours, stage-specific alert types, pre-filter scores |
| **F3 evasive attacker** | low-rate attacker keeps its flow rate under a level that produces ≤ 2 alerts per 10 s per gateway and moves to fresh gateways once its alerts exceed that | an attack shows up as excess alerts that accumulate | alerts, pre-filter scores, flow sizes |
| **F4 benign surges** | flash crowds: benign flows ×5–×10 on random gateways for 5–15 s, some routed to the flood stream by the header pass, raising false alerts | excess alerts mean attacks | flow and packet counts, packet sizes, stream, alerts |
| **F5 infrastructure** | other trees (fan-out 2–8, fog capacity ×0.5–×2, uplink ×0.25–×1) and loads x1–x5 | — (values keep their meaning; which value errors cost decisions changes) | as now |

## Competitors (same annealer, budget and repair; only the value estimate differs)

1. **fixed** — the current formula, parameters set once on x3.
2. **adaptive (L1+L2)** — the same formula with an online false-positive rate per gateway
   (EWMA over epochs where the gateway's pre-filter scores are low), recall updated from
   delayed labels (Beta posterior per class, labels 30 s late), and a Page–Hinkley test on
   each gateway's alert rate that resets the 40 s window.
3. **HMM (L3, optional)** — per gateway two-state hidden Markov model of "attack present"
   with emissions from the current rates; only if a learned model beats (2).
4. **two-stage ML** — gradient-boosted trees (and a small MLP) from task features to the
   exact value; squared-error loss.
5. **decision-focused ML** — same model class trained on decision quality: SPO+ loss with
   the fast annealer as the solver (the epoch objective is linear in the values for fixed
   load terms); perturbed Fenchel–Young loss as a second variant.
6. **oracle values** — exact per-task values from ground truth (`ps-anneal-oracle-value`).

Task features (all visible at decision time): pre-filter score, stream, flows, packets,
bytes, deadline class, gateway criticality, alerts / expected false alerts / analysed
recall over 10, 20 and 40 s (as recorded by `scripts/collect_values.py`), the same for
the gateway's fog-subtree neighbours, the gateway's device mix (F1). Inference has to fit
in 2 ms per epoch.

## Protocol

- Training data: seeds 7–26 per family; evaluation seeds 2–6 in the Python simulator and,
  for the final table in iFogSim, seeds 2–6 and 27–41 (disjoint from training; the earlier
  20-seed tables used 2–21, which overlap the training seeds).
- **Leave-one-family-out**: models are trained on four of F1–F5 (plus F0) and evaluated on
  the held-out family; also reported: trained on all families.
- Metrics: objective (macro missed detection + 0.05 × false alerts per benign flow),
  missed low-rate and flood tasks, decision time; prediction quality (Spearman correlation
  and error against exact values); share of the headroom closed,
  (fixed − method) / (fixed − oracle).
- Paired differences over seeds with 95 % Student-t intervals.

## Hypotheses

- **H1** Headroom (fixed − oracle) is at least 3 points in F2, F3 and F4.
- **H2** The adaptive formula closes at least half of the headroom in F1 and less than a
  third in F2 and F4.
- **H3** Two-stage ML, on a held-out family, beats the adaptive formula by at least
  1 point in F2 and F4.
- **H4** Decision-focused training is at least as good as two-stage training in at least
  four of the six families, with the largest gain in F5 and at low load.
- **H5** Somewhere (expected: low load, F5) decision-focused values beat the oracle values,
  i.e. accurate values are not the best coefficients.
- **H6** In F3 every method stays far from the oracle: an attacker that keeps its evidence
  below the noise cannot be found from alerts, learned or not.

If H3 fails, the paper reports that online statistics are enough for these families; if
H4 fails, that value accuracy is the right target here.

## Work order

1. Generator families F1–F5 (new config blocks; existing configs, tests and the pinned
   small-instance checksum unchanged); headroom per family (fixed vs oracle).
2. Adaptive formula (L1+L2).
3. Data collection per family; two-stage models.
4. Decision-focused training.
5. Final table in iFogSim.

## Amendments (before the family experiments)

- **F3** is implemented with a fixed migration schedule (every 8–12 s to gateways not used
  in the last two periods) instead of an attacker that reacts to its own alerts: the
  traffic of a run is generated before the run starts. The rate (0.25 flows/s per class
  and gateway) gives about two alerts per 10 s on a fully analysed gateway.
- **Features for the families** (feature set v2, `fogids.learned.FEATURES_V2`): bytes per
  packet, the gateway's device profile (F1; device types are known to the operator) and
  the alert evidence of the other gateways of the same fog subtree over 40 s.
- Results so far on the earlier conditions (x1, x3, x5, 5 % separation error, drift) are in
  [smokescreen.md](smokescreen.md): the learned estimate beats the hand-tuned formula by
  0.7–1.0 points and decision-focused weights add 0.2–0.3 points (iFogSim, 20 seeds).
