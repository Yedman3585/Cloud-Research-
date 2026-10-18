# Scenario v2: a Flood as a Smokescreen (screening)

Code: [`smokescreen.py`](../src/fogids/smokescreen.py), tests
[`test_smokescreen.py`](../tests/test_smokescreen.py), screening
[`scripts/screen_smokescreen.py`](../scripts/screen_smokescreen.py), configs
`configs/ids-place-smokescreen.json` (x1) and `-x3.json`.

## Why

On the calibrated scenario, simple rules are already within 0.2–0.3 points of the
fluid bound (`ps-model.md`), so no method can win much there. Two measurements make a
harder scenario realistic rather than tuned (`calibration.md`):

- feature extraction costs 7–8 µs per packet on an M2 (dpkt), 6–60 times the inference
  of a record; a DDoS burst offers a Pi-class gateway about eleven times the packets
  it can parse, so an IDS under a flood must analyse only a share of the traffic;
- with only every s-th flow analysed, the context model keeps its per-flow quality, so
  sampling acts through coverage only.

Attackers exploit exactly this: a flood saturates the monitor while a low-rate attack
(reconnaissance, brute force, web) slips through.

## Model

Changes with respect to IDS-PLACE (details in the module docstring):

- a cheap header pass splits every window into a *flood* stream (heavy hitters) and a
  *residual* stream, with 2 % errors both ways; each stream is cut into tasks;
- extraction is CPU work: 0.06 MI per analysed packet (7.5 µs at 8000 MIPS) plus
  0.0004 MI per packet for the header pass; 10 packets per benign or low-rate flow,
  100 per flood flow (CICIoT2023 `Number`);
- a decision is (node, model, fraction f ∈ {1, 0.5, 0.2, 0.05, 0.01}, extraction
  site): features extracted at the gateway and shipped, or sampled raw packets shipped
  and extracted at the fog or cloud node; f = 0 drops the task;
- a task containing a_c flows of class c is detected with probability
  1 − (1 − f r_c)^(a_c) if it finishes by its deadline; analysed benign flows raise
  false alerts at the model's rate;
- objective: macro-averaged missed detection over attack classes, plus 0.05 × false
  alerts per benign flow.

Traffic (medium topology: 4 fogs × 8 Pi-class gateways): benign 300 flows/s per
gateway; a DDoS flood on 12 gateways (15–40 s) and a DoS flood on 8 others (42–54 s) at
CICIoT2023 class ratios; low-rate recon (40 flows/s), web and brute force (15 flows/s)
on 6 gateways each, 10–56 s, half of them under a flood. The x3 variant triples the
benign and flood rates; low-rate attacks stay low-rate.

The simulator is event-driven with processor-sharing CPUs and FIFO uplinks, the
semantics of the iFogSim driver (`ps-model.md`); it is checked against the
processor-sharing closed form. The bound is the fluid LP of `bounds.py` over all
options, solved per fog subtree (cloud CPU unconstrained) or, for x3, per gateway (only
the gateway's CPU and uplink constrained); both are valid lower bounds, the second
looser.

## Reference policies

| Policy | Rule |
|---|---|
| `no-shed` | f = 1, features at the gateway, risk-split placement |
| `uniform-shed` | classic load shedding: per gateway and epoch, the largest common f with which every task is predicted to meet its deadline |
| `greedy-v2` | `ps-greedy-s0.1` over the new options: highest risk first, largest f, then best model, earliest predicted finish; else drop |
| `stream-aware` | expert rule against smokescreens: flood tasks at the smallest f with the light model at the gateway, residual tasks as `greedy-v2` |
| `stream-aware-forecast` | as `stream-aware`, plus the gateway's next window (assumed equal to this one) as phantom work |
| `value-greedy` | myopic value maximiser: best estimated detection value per gateway CPU |

## Screening, seed 1

| Policy | x1: objective | x1: missed low-rate / flood | x3: objective | x3: missed low-rate / flood |
|---|---|---|---|---|
| fluid bound | **0.002** (fog subtree) | 0.001 / 0.000 | **0.046** (per gateway) | 0.005 / 0.102 |
| `no-shed` | 0.711 | 0.526 / 0.980 | — | — |
| `uniform-shed` | 0.212 | 0.289 / 0.084 | 0.652 | 0.429 / 0.975 |
| `greedy-v2` | 0.147 | 0.046 / 0.284 | **0.370** | 0.151 / 0.684 |
| `stream-aware` | **0.014** | 0.009 / 0.006 | 0.488 | 0.209 / 0.893 |
| `stream-aware-forecast` | — | — | 0.487 | 0.207 / 0.893 |
| `value-greedy` | 0.414 | 0.271 / 0.625 | 0.605 | 0.541 / 0.699 |

Reading:

1. **x1 fails the screening criterion.** Generic rules are far from the bound, but the
   expert rule `stream-aware` is within 1.2 points of it: once heavy hitters are
   separated, sampling the flood to 1 % and analysing the rest fully is nearly
   optimal. A new method could not beat that rule by a meaningful margin.
2. **x3 passes, with a caveat.** Every rule misses 15–54 % of low-rate attack tasks and
   68–98 % of flood tasks; the best (`greedy-v2`, 0.370) is 32 points above the bound.
   The expert rule breaks: all flood chunks of a window arrive together, and under
   processor sharing even at f = 1 % (40 MI header pass + 60 MI extraction each)
   fifteen of them finish together after about 0.6 s, past their 0.3 s deadline. The
   persistence forecast does not help because the failure is within the window, not
   across windows.
3. **The bound is optimistic here.** The fluid LP ignores processor sharing (it can
   serve work in any order) and, for x3, the fog's capacity. Under processor sharing,
   meeting the deadlines needs a split of each window's flood chunks between
   extraction at the gateway, raw-packet offload to the fog (FIFO uplink) and dropping,
   with the residual tasks fitted in between: a coupled assignment whose exact cost is
   the processor-sharing min-form of the epoch QUBO. How much of the 32 points is
   attainable is not known yet; the epoch QUBO on x3 is the test.

## Held-out seeds 2–6

Same configurations, seeds 2–6 (the scenario was designed on seed 1); bound per gateway
(only the gateway's CPU and uplink constrained) for both loads; mean ± 95 % Student-t
half-width over seeds. Missed shares are per task containing the class.

| Policy | x1: objective | x1: missed low-rate / flood | x3: objective | x3: missed low-rate / flood |
|---|---|---|---|---|
| fluid bound (per gateway) | **0.002 ± 0.000** | 0.001 / 0.000 | **0.047 ± 0.001** | 0.005 / 0.105 |
| `no-shed` | 0.714 ± 0.003 | 0.528 / 0.982 | — | — |
| `uniform-shed` | 0.213 ± 0.001 | 0.289 / 0.085 | 0.648 ± 0.005 | 0.421 / 0.977 |
| `greedy-v2` | 0.167 ± 0.012 | 0.051 / 0.326 | **0.386 ± 0.007** | 0.163 / 0.708 |
| `stream-aware` | **0.010 ± 0.001** | 0.004 / 0.003 | 0.488 ± 0.010 | 0.208 / 0.896 |
| `stream-aware-forecast` | 0.010 ± 0.001 | 0.004 / 0.003 | 0.489 ± 0.011 | 0.208 / 0.896 |
| `value-greedy` | 0.402 ± 0.009 | 0.272 / 0.594 | 0.594 ± 0.013 | 0.514 / 0.712 |

The seed-1 picture holds on every held-out seed, with the same ranking: at x1 the
expert rule is within 0.8 points of the bound; at x3 the best rule (`greedy-v2`) is
33.9 points above it, missing 16 % of low-rate attack tasks and 71 % of flood tasks,
and the persistence forecast changes nothing. The x3 scenario is the one on which the
epoch QUBO has to show what part of that gap is attainable under processor sharing.

## Joint epoch optimisation (x3)

Code: [`smokescreen_qubo.py`](../src/fogids/smokescreen_qubo.py), tests
[`test_smokescreen_qubo.py`](../tests/test_smokescreen_qubo.py).

Two epoch models, both annealed for 50 ms per epoch and checked and repaired against
the processor-sharing estimator:

- `qubo-smoke`: the qubo-ps energy extended to the new options (value of an option
  minus the processor-sharing sum of deadline-normalised finish times). It was worse
  than `greedy-v2` on seed 1 for every weight tried (0.43–0.56): a finish-time penalty
  charges every job, not only those that miss, so it either drops floods or starves the
  residual stream.
- `ps-anneal`: the same processor-sharing rows used as **deadline constraints**. A job's
  predicted latency is the sum over its stages of ``sum_p min(a_o, a_p)`` (the QUBO
  row); it earns its value only if that fits its budget, and lateness is penalised
  (kappa = 2). Jobs already committed are fixed members of the model. Parameters chosen
  on seed 1 (kappa ∈ {2, 5}, prior of 3, 5 or 10 attack flows per task).

Option values are scheduler-visible: detection probability with the attack-flow count
estimated from the pre-filter score (at least 5 flows, since a low-rate attack hides
below the score's noise), the value of a flood window shared among its chunks. As a
reference, `ps-anneal-oracle-value` uses each option's exact contribution to the
objective (ground truth), everything else unchanged.

Seeds 2–6, x3, means:

| Policy | Objective | Missed low-rate | Missed flood | Decision, ms |
|---|---|---|---|---|
| fluid bound (per gateway) | 0.047 | 0.005 | 0.105 | — |
| `greedy-v2` | 0.386 | 0.163 | 0.708 | — |
| `ps-anneal` | **0.366** | 0.235 | **0.550** | 56 |
| `ps-anneal-oracle-value` | **0.290** | 0.110 | 0.555 | 57 |

Paired against `greedy-v2`, `ps-anneal` is better on every seed (−2.0 points on
average, from −0.1 to −4.3). Reading:

1. **Joint placement under processor sharing works on the part it can see.** It detects
   16 points more flood tasks than the greedy rule: the window's chunks are split
   between gateway extraction, raw offload and dropping so that the kept ones meet
   their deadlines.
2. **It loses on low-rate attacks because their value is invisible.** With a pre-filter
   score dominated by noise, a residual task with a low-rate attack looks like a benign
   one; the annealer samples it down where the greedy rule, which always tries f = 1
   first, keeps it whole.
3. **With the true values the same optimiser reaches 0.290**, 9.6 points better than
   `greedy-v2`, and misses 11 % of low-rate tasks instead of 16–24 %. The gap between
   `ps-anneal` and its oracle (7.6 points) is what a better value estimate can win; the
   gap between the oracle and the fluid bound is mostly the bound's optimism (it ignores
   processor sharing: flood detection stays at 0.55 even with true values).

This locates the learning problem for the method: not the load of future epochs (the
forecast did not help), but the **value coefficients of the epoch model** — which tasks
are worth analysing — predicted from scheduler-visible signals such as recent alerts of
the same gateway, and trained decision-focused through the annealer.

## Next

- Run the epoch QUBO (extended with fraction and extraction site, exact
  processor-sharing terms) on x3; a tighter bound that respects processor sharing.
- More seeds, after the QUBO run, to quantify the gaps.
- The oracle-future test (QUBO given the true next windows) on x3, to measure what a
  learned forecast could add.
