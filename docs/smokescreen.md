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

The gap between `ps-anneal` and its oracle is therefore informational: the scheduler
needs a better estimate of which tasks are worth analysing. The next section shows
that the IDS's own alerts provide it.

## Alert feedback (x3)

The value of a residual task depends on whether it hides a low-rate attack, which the
pre-filter score cannot tell. An IDS does report something the scheduler can use: the
alerts it raised in earlier windows. The simulator now samples them for every task
analysed on time (`Engine._alert`): attack flows of the analysed share are reported in
their group (flood or low-rate) with the model's recall; analysed benign flows raise a
low-rate alert at the model's false-positive rate. That every benign misclassification
of both models falls in the low-rate group is measured on CICIoT2023
(`scripts/alert_groups.py`: 25.6 % of benign records for `light`, 12.6 % for `full`;
flood classes are never reported as low-rate).

`AlertEvidence` turns the residual-stream alerts of a gateway over the last 40 s into
an estimated share of low-rate attack flows: (alerts − false alerts expected if all
analysed flows were benign) / (analysed flows × mean recall). As a ranking of residual
tasks with and without a low-rate attack (seed 1, decisions of `greedy-v2`) it reaches
an AUC of 0.76, against 0.64 for the pre-filter score (0.65 with a 5 s history, 0.73
with 20 s).

Policies using it: `ps-anneal-alerts` (attack flows of a residual task = max(3, share ×
flows); floor and horizon chosen on seed 1) and, as a fair baseline with the same
information, `value-greedy-alerts`. Seeds 2–6, x3:

| Policy | Objective | Missed low-rate | Missed flood |
|---|---|---|---|
| `greedy-v2` | 0.386 | 0.163 | 0.708 |
| `value-greedy-alerts` | 0.321 | 0.144 | 0.575 |
| `ps-anneal` | 0.366 | 0.235 | 0.550 |
| **`ps-anneal-alerts`** | **0.289** | **0.086** | 0.578 |
| `ps-anneal-oracle-value` | 0.290 | 0.110 | 0.555 |

1. **Alert feedback plus joint placement gives the best policy: 0.289**, 9.7 points
   below `greedy-v2` and 3.2 points below the greedy rule with the same alert
   information (better on every seed, by 1.9 to 4.8 points).
2. **The simple statistical estimate already matches the oracle values** (0.289 against
   0.290 at the same 50 ms budget). In this scenario a learned value model has nothing
   left to win; what remains is the optimiser and the flood capacity.
3. **The optimiser is the lever now.** With the oracle values, three times the annealing
   budget (150 ms) gives 0.249 on seed 1 instead of 0.288: a faster annealer reaches
   more of it within the 100 ms epoch.

## Faster annealer

`fast_anneal.py` runs the same `EpochPS` model and moves on dense arrays compiled with
numba: a move updates the latency of every job sharing a resource with the moved task
by min(a_j, a_k), so it costs O(jobs on the resources involved). About 2.5 million
moves per second against about 20 thousand in pure Python, i.e. ~120 000 moves in the
50 ms budget instead of ~1 000. Dominated options (no more value, at least as much load
on every resource) are removed before annealing (`_pareto`; the processor-sharing and
FIFO delays are monotone in every load, so this is exact). The kernel's energy is
checked against the Python model in the tests. Decision time per epoch, including
building the model and the repair pass: 65–70 ms of the 100 ms epoch.

Seeds 2–6, x3, Python simulator: `ps-anneal` 0.366 → 0.341, `ps-anneal-alerts` 0.289 →
0.256, `ps-anneal-oracle-value` 0.290 → 0.230. With the oracle values the optimiser now
leaves the alert-based policy 2.6 points behind: the value estimate matters again once
the search is good enough.

## Final comparison in iFogSim (x3)

`org.fogids.IdsStaged` executes the staged tasks in iFogSim: a CPU stage on the gateway,
then optionally a second CPU stage on the gateway or an ancestor, reached over the
uplinks with the given bytes; work in MI and node MIPS are both scaled by 1000 (tuple
lengths are integers) and the CPU update interval is 1 ms (deadlines of 0.3 s on
tightly packed gateways). `smokescreen_ifogsim.simulate_ifogsim` runs a policy against
it: the policy sees a shadow `Engine` driven by the same decisions and the alerts of
the tasks iFogSim completed on time; metrics use iFogSim's finish times. On the test
scenario iFogSim reproduces the Python simulator's finish times within 3 ms and the same
objective.

Seeds 2–6, x3, metrics from iFogSim (mean ± 95 % half-width; differences paired by
seed):

| Policy | Objective | Missed low-rate | Missed flood | Decision, ms |
|---|---|---|---|---|
| `greedy-v2` | 0.393 ± 0.008 | 0.163 | 0.724 | 41 |
| `value-greedy-alerts` | 0.330 ± 0.017 | 0.144 | 0.597 | 70 |
| `ps-anneal` | 0.341 ± 0.011 | 0.217 | 0.518 | 68 |
| **`ps-anneal-alerts`** | **0.256 ± 0.008** | **0.019** | 0.596 | 67 |
| `ps-anneal-oracle-value` (Python simulator) | 0.230 | 0.021 | 0.542 | 49 |

`ps-anneal-alerts` is 13.7 points below `greedy-v2` and 7.5 points below the greedy
rule with the same alert information, better on every seed (by at least 5.7 points),
and misses 1.9 % of low-rate attack tasks instead of 14–16 %. The scheduler's own
prediction (shadow) of these runs is within 0.001 of iFogSim's objective; finish times
differ by 6 ms on average.

## Robustness and ablations (Python simulator, seeds 2–6)

`scripts/robustness.py` changes one parameter of the x3 scenario at a time, or one
component of `ps-anneal-alerts`; 165 runs. Objective, mean ± 95 % half-width; the last
column is paired by seed.

| Variant | `greedy-v2` | `value-greedy-alerts` | `ps-anneal-alerts` | `ps-anneal-alerts` − `value-greedy-alerts` |
|---|---|---|---|---|
| x3 (base) | 0.386 ± 0.007 | 0.321 ± 0.013 | **0.255 ± 0.008** | −0.066 ± 0.019 |
| load x2 | 0.309 ± 0.013 | 0.252 ± 0.007 | **0.149 ± 0.010** | −0.103 ± 0.010 |
| load x5 | 0.456 ± 0.015 | 0.412 ± 0.010 | **0.379 ± 0.007** | −0.033 ± 0.009 |
| extraction 1 µs/packet | 0.408 ± 0.016 | 0.323 ± 0.019 | **0.088 ± 0.010** | −0.235 ± 0.019 |
| extraction 3 µs/packet | 0.411 ± 0.012 | 0.347 ± 0.010 | **0.154 ± 0.009** | −0.193 ± 0.006 |
| separation error 0 % | 0.353 ± 0.005 | 0.285 ± 0.014 | **0.251 ± 0.008** | −0.034 ± 0.015 |
| separation error 5 % | 0.497 ± 0.015 | 0.421 ± 0.016 | **0.350 ± 0.010** | −0.071 ± 0.016 |
| large topology (8 × 16 gateways) | 0.378 ± 0.006 | 0.315 ± 0.005 | **0.260 ± 0.003** | −0.055 ± 0.005 |

`ps-anneal-alerts` is the best policy in every variant and on every seed; its margin
over `greedy-v2` ranges from 5.0 points (worst seed, x5) to 32 points (1 µs per packet).
The margin is largest when extraction is cheaper (a compiled extractor): the greedy
rules then spend the freed CPU on flood chunks first, while the epoch model puts it on
the residual stream. It shrinks at x5, where every policy is capacity-bound, and when
the streams separate perfectly (the low-rate attacks are then easy for everyone).

Ablations (x3):

| Change | Objective | Missed low-rate | Missed flood | Decision, ms |
|---|---|---|---|---|
| none | 0.255 ± 0.008 | 0.018 | 0.597 | 58 |
| no alerts (`ps-anneal`) | 0.341 ± 0.010 | 0.216 | 0.519 | 60 |
| no dominance pruning | 0.256 ± 0.008 | 0.018 | 0.598 | 55 |
| no repair | 0.256 ± 0.008 | 0.018 | 0.599 | 58 |
| annealing budget 10 ms | 0.258 ± 0.008 | 0.022 | 0.597 | 16 |
| annealing budget 25 ms | 0.257 ± 0.007 | 0.021 | 0.597 | 32 |
| annealing budget 100 ms | 0.255 ± 0.008 | 0.016 | 0.599 | 111 |
| alert window 10 s | 0.255 ± 0.009 | 0.018 | 0.596 | 58 |
| alert window 20 s | 0.257 ± 0.007 | 0.019 | 0.599 | 58 |
| oracle values | 0.229 ± 0.005 | 0.018 | 0.540 | 42 |

The two components that matter are the alert-based values (8.6 points) and the
deadline-exact epoch model (the step from `value-greedy-alerts`, 6.6 points). Pruning,
repair, the alert window and the annealing budget change less than the seed-to-seed
noise: with the compiled annealer, 10 ms per epoch already reaches the result, so the
method needs about 16 ms of the 100 ms epoch. What remains to the oracle (2.6 points)
is on flood tasks (0.597 against 0.540 missed), not low-rate ones.

## Next

- A bound that respects processor sharing, so the remaining gap (to 0.230 with oracle
  values, to 0.047 for the fluid bound) can be stated against an attainable reference.
- More scenarios for generality: x1 and an intermediate load, other topologies, a
  different flood/low-rate mix.
- Whether a learned value estimate closes the 2.6 points between the alert-based
  estimate and the oracle values.
