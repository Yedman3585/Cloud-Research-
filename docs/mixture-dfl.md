# Per-family value weights: a routed mixture against the adaptive formula

Plan and families: [learning-plan.md](learning-plan.md), [families.md](families.md).
This note asks a narrower question than the plan: if the decision-focused value weights
(`fogids.dfl`) are specialised per family and a perfect router picks the matching set at
run time, do they beat a single set of weights trained on all families, and do they beat
the online adaptive formula (`fogids.adaptive`)?

The router is given the true family (oracle routing). It is therefore an upper bound on
any real context classifier: if the oracle-routed mixture does not beat the pooled weights,
no classifier over the same features can.

## Setup

A reduced rerun, meant to establish direction rather than final figures; it should be
repeated at the full budget before the numbers are quoted as results. Differences from
the main runs:

- value data from seeds 7–12 only (six per family), `scripts/collect_values.py --variant <family>`;
- one shared attack-flow estimator for every policy, `scripts/train_values.py --features v2`
  over the five families (so only the weights differ between the DFL policies);
- weights trained with the zeroth-order regret method at a reduced budget
  (`--steps 40 --pairs 6 --batch 80 --every 4`), one set per family and one set over all five;
- evaluation on seeds 2–6, `scripts/eval_mixture.py`.

Policies, all sharing `value-gbm-v2`:

| Name | Value weights |
|---|---|
| alerts | fixed formula, false-alert weight 1e-3 (`ps-anneal-alerts`) |
| adaptive | formula with online false-positive rate and change detection (`ps-anneal-adaptive`) |
| dfl-pooled | one set of weights trained on all five families |
| dfl-expert | the set trained only on the evaluated family (oracle routing) |
| oracle-value | exact ground-truth values in the same annealer (upper bound) |

## Result

Objective, lower is better, mean ± 95 % interval over seeds 2–6.

| Family | alerts | adaptive | dfl-pooled | dfl-expert | oracle-value |
|---|---|---|---|---|---|
| F1 detectors | 0.350 ± 0.043 | 0.281 ± 0.007 | 0.272 ± 0.018 | 0.270 ± 0.017 | 0.238 ± 0.008 |
| F2 botnet | 0.358 ± 0.028 | 0.366 ± 0.022 | 0.358 ± 0.029 | 0.354 ± 0.029 | 0.314 ± 0.032 |
| F3 evasive | 0.314 ± 0.016 | 0.406 ± 0.022 | 0.326 ± 0.023 | 0.319 ± 0.017 | 0.262 ± 0.011 |
| F4 surges | 0.380 ± 0.017 | 0.354 ± 0.017 | 0.374 ± 0.020 | 0.370 ± 0.022 | 0.335 ± 0.011 |
| F5 infra | 0.159 ± 0.174 | 0.157 ± 0.167 | 0.155 ± 0.159 | 0.159 ± 0.157 | 0.153 ± 0.134 |

Expert minus adaptive (negative means the weights win): F1 −0.011, F2 −0.012,
**F3 −0.087**, F4 +0.016, F5 +0.001.

## Reading

**The weights are robust where the adaptive formula fails.** On F3 the adaptive formula is
the worst of all policies, 0.406 against 0.314 for the fixed formula. The low-rate
attackers stay under about two alerts per 10 s and move every 8–12 s, so the change
detector fires on noise, the window is repeatedly cut to 5 s, evidence never accumulates,
and the policy misses more low-rate attacks than any other — missed low-rate 0.272, against
0.127 for the expert and 0.120 for the fixed formula. Both the pooled and the expert weights
avoid this because they do not re-estimate the false-positive rate online. On the family
built to suppress the alert signal, the learned weights are the safe choice.

**Specialising per family buys almost nothing over pooling.** The oracle-routed expert and
the single pooled set differ by at most 0.007 on every family, inside the intervals. The
task features already carry the context into one model, so a router over those features —
even a perfect one — has no headroom to exploit. A context classifier and a bank of
per-family weights are not justified by these numbers; one pooled set is enough.

**Neither online nor offline dominates across families.** The adaptive formula wins where a
rate genuinely shifts (F4 surges, 0.354 against 0.370 for the weights; F1 close) and loses
where the signal is suppressed (F3). The weights are even or ahead everywhere except F4 and
never fail the way the adaptive formula does on F3. The gap to exact values stays 2–9 points
on every family and policy.

## Takeaway

The pooled decision-focused weights are not a method that needs a router to be useful; they
are the robust fallback when the alert signal is adversarially suppressed, and even with the
rest of the families. The per-family mixture, at its oracle upper bound, does not improve on
them, so the routing idea is not worth building here.
