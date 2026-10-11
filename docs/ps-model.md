# Processor-Sharing Model and `ps-greedy`

Code: [`ps_model.py`](../src/fogids/ps_model.py), [`ps_policies.py`](../src/fogids/ps_policies.py),
tests [`test_ps_model.py`](../tests/test_ps_model.py), evaluation
[`scripts/ps_eval.py`](../scripts/ps_eval.py).

## Why

M5 (`ids-place.md` §9) showed that the scheduler-side estimate of the queue-aware
baselines (`QueueModel`) fails under load: its latency ranking is uncorrelated with the
simulator at x3 and inversely correlated at x10, and every deadline miss of
`queue-greedy` was predicted to meet. `QueueModel` treats a node's CPU as a FIFO
backlog. The simulator does something else.

## What the simulator does

CloudSim time sharing on a single-PE node is **egalitarian processor sharing**: n
running tasks each progress at MIPS / n, so a task that arrives later slows every task
already running, and a short late task can finish before a long early one. Checked
directly: on a 1000-MIPS node, a 1000-MI task at t = 1.0 and a 200-MI task at t = 1.1
finish at 2.2 and 1.5 s (processor sharing), not 2.0 and 2.2 s (FIFO). Uplinks are FIFO:
a transfer occupies the link for bytes / bandwidth, then the propagation latency is
added.

## The model

`PSModel` keeps, per node, the tasks the policy has committed there (running ones with
remaining work, and ones still in transit with their arrival time) and evolves them with
the same processor-sharing rule; uplinks are tracked as FIFO queues; resident models
follow the simulator's least-recently-used rule within node memory. For a candidate
(task, node, model) it returns the task's predicted finish time and the new finish
times of every task already on that node. It is synchronised from the snapshot
(completions, resident models) and knows nothing about tasks of future epochs.

On a hand-built case with local and fog execution, shared CPUs and staggered arrivals
it reproduces the simulated finish times within the CPU update interval (test
`test_model_predicts_simulated_finish_times`).

**Closed form.** For tasks that start together on one node with works a_i, the sum of
processor-sharing finish times (in units of work / MIPS) is

    sum_i a_i + 2 sum_{i<j} min(a_i, a_j),

so with binary placement variables y_i the node cost is
`sum_i a_i y_i + 2 sum_{i<j} min(a_i, a_j) y_i y_j`, an **exact quadratic (QUBO) term**.
Work of tasks already running adds only linear terms. The tests check the formula
against simulation for random works and as a quadratic over every subset of tasks.
This is the term an epoch QUBO for IDS-PLACE should use for CPU contention.

## `ps-greedy`

Tasks in order of decreasing risk weight. For each task, among its eligible nodes and
the models in quality order, take the best model whose predicted finish meets the
deadline **and** which pushes no task already on that node, predicted to meet its own
deadline, past it. If no option qualifies, take the one creating the fewest new misses,
then finishing first.

Without further care this does not help: models other than the fastest end up committed
with no slack, and a task of a later epoch, which the model cannot foresee, then breaks
them. `ps-greedy-s0.1` therefore asks a model other than the fastest to finish within 90 %
of the deadline. The slack was chosen on seed 1 (sweep 0, 0.1, 0.2, 0.3, 0.5 at x1, x3,
x10; 0.1 is the smallest value with almost no misses) and is evaluated below on seeds
2–6 only.

## Results

Medium instance, calibrated models, seeds 2–6 (held out), decision time not charged; mean
± 95 % Student-t half-width. "Estimate" compares the latency each policy predicted at
decision time with the simulated one on completed tasks; for `risk-split` and
`queue-greedy` the decisions are also replayed through `PSModel`, so both estimators are
scored on identical decisions.

| Load | Policy | Miss rate | False alerts | Estimate Spearman (own / PSModel) | Estimate MAE, ms (own / PSModel) | Decision, ms |
|---|---|---|---|---|---|---|
| x1 | fluid bound | 0 | 0.1268 | | | |
| x1 | risk-split | 0 | 0.1453 | 1.00 / 1.00 | 1 / 1 | 0.2 |
| x1 | queue-greedy | 0 | 0.1268 | 0.92 / 0.92 | 21 / 21 | 0.3 |
| x1 | ps-greedy-s0.1 | 0 | 0.1268 | — / 0.92 | — / 21 | 0.4 |
| x3 | fluid bound | 0 | 0.1308 | | | |
| x3 | risk-split | 0 | 0.1454 | 0.96 / 0.96 | 7 / 7 | 0.3 |
| x3 | queue-greedy | 0.052 ± 0.012 | 0.1332 | **0.07 / 0.89** | **87 / 29** | 0.5 |
| x3 | ps-greedy | 0.064 ± 0.011 | 0.1332 | — / 0.89 | — / 29 | 1.2 |
| x3 | ps-greedy-s0.1 | **0.0002 ± 0.0005** | **0.1340** | — / 0.90 | — / 24 | 1.2 |
| x10 | fluid bound | 0 | 0.1374 | | | |
| x10 | risk-split | 0 | 0.1453 | 0.89 / 0.89 | 20 / 20 | 1.3 |
| x10 | queue-greedy | 0.055 ± 0.005 | 0.1387 | **−0.57 / 0.87** | **136 / 29** | 1.8 |
| x10 | ps-greedy | 0.187 ± 0.027 | 0.1394 | — / 0.91 | — / 28 | 26.5 |
| x10 | ps-greedy-s0.1 | **0.0027 ± 0.0018** | **0.1393** | — / 0.87 | — / 27 | 4.7 |

Reading:

1. **The processor-sharing model fixes the estimate.** On the same decisions of
   `queue-greedy`, the rank correlation with the simulator goes from 0.07 to 0.89 at x3
   and from −0.57 to 0.87 at x10, and the mean absolute error falls from 87 and 136 ms
   to 29 ms. The remaining error comes from tasks of future epochs.
2. **An accurate estimate alone does not remove misses.** Broken promises on
   `queue-greedy`'s decisions are about 5 % for both estimators: they are caused by later
   arrivals. `ps-greedy` without slack misses as often (x3) or more (x10).
3. **With a 10 % slack, ps-greedy keeps queue-greedy's alert quality without its misses.**
   At x3: 0.02 % misses and 0.1340 false alerts against 0 and 0.1454 for `risk-split`,
   i.e. 78 % of the gap between `risk-split` and the fluid bound (0.1308) is closed. At
   x10: 0.27 % misses and 0.1393 against 0 and 0.1453, 76 % of the gap to 0.1374.
4. **The absolute gain is small** (1.1 and 0.6 points of false alerts) because the
   calibrated scenario leaves little headroom (§9 of `ids-place.md`); the gain of an
   accurate model will be larger only in scenarios where simple rules fall further short.
5. Decision time stays far below the 100 ms epoch (4.7 ms mean at x10).

Reproduce:

```bash
python scripts/ps_eval.py --seeds 2 3 4 5 6 \
  --config configs/ids-place-medium-calibrated.json \
  --config configs/ids-place-medium-calibrated-x3.json \
  --config configs/ids-place-medium-calibrated-x10.json
```

`ps-greedy` is not yet registered in `policies.by_name`, so `fogids simulate`, `sweep`
and `oracle` do not list it; `scripts/ps_eval.py` imports it directly.

## Epoch QUBO (`qubo-ps`)

Code: [`qubo_place.py`](../src/fogids/qubo_place.py), tests
[`test_qubo_place.py`](../tests/test_qubo_place.py).

At every epoch the released tasks are placed jointly. Variables x_o = 1 if task i takes
option o = (node, model), one option per task. Energy:

- false alerts: `lam · benign_i · fpr_k` (benign flows estimated from the pre-filter);
- CPU, exact for processor sharing of tasks starting together:
  `gamma / MIPS · [Σ w_o a_o x_o + Σ_{o<p, same node} (w_o + w_p) min(a_o, a_p) x_o x_p]`,
  plus linear terms `(w_o + w_j) min(a_o, r_j)` for every task j already on the node
  (remaining work r_j from `PSModel`);
- uplinks: current wait and transfer time of every link on the path (linear) and
  `(w_o + w_p) min(b_o, b_p) / bandwidth` for two new tasks on the same link;
- one option per task: `P (Σ x - 1)²` in the plain QUBO (`EpochQubo.to_qubo`); the
  annealer moves between the options of one task, which keeps the constraint exactly.

Weights w are 1 / deadline (normalised latency); optionally `1 / (margin + u)` with the
predicted margin to the deadline (`urgency=u`), which makes latency expensive only where
deadlines are tight. Options that cannot meet their deadline under the present load are
dropped before the QUBO is built (the fastest model at the gateway is always kept), the
QUBO is annealed for 20 ms, and the decoded assignment is checked task by task against
`PSModel` (own deadline with the 10 % slack, no committed task pushed past its
deadline); a failing task takes the `ps-greedy` choice.

Tests: the CPU coefficients reproduce the processor-sharing sum of finish times; the
plain QUBO equals the one-hot energy on every feasible assignment and penalises the
empty one; the annealer finds the exhaustive optimum on a small epoch; a simulated run
places every task once.

`gamma` (latency weight) was chosen on seed 1 among 0, 1, 10, 100, 1000: 10 is the
smallest value at which most annealed assignments pass the check unchanged (4.5 %
repaired at x3); 0 reduces the QUBO to a false-alert minimiser that relies on repair
(18 %), large values push work to the light model. Seeds 2–6:

| Load | Policy | Miss rate | False alerts | Decision, ms |
|---|---|---|---|---|
| x1 | ps-greedy-s0.1 | 0 | 0.1268 | 0.5 |
| x1 | qubo-ps | 0 | 0.1323 ± 0.0002 | 21.7 |
| x3 | ps-greedy-s0.1 | 0.0002 ± 0.0005 | 0.1340 ± 0.0001 | 1.2 |
| x3 | qubo-ps | **0** | 0.1349 ± 0.0001 | 25.6 |
| x10 | ps-greedy-s0.1 | 0.0027 ± 0.0018 | 0.1393 ± 0.0006 | 4.6 |
| x10 | qubo-ps | 0.0016 ± 0.0014 | 0.1399 ± 0.0006 | 53.6 |

Reading: on the calibrated scenario the epoch QUBO is **on par with `ps-greedy-s0.1`,
not better**. It removes the last misses at x3 and x10 at the cost of 0.1 point of false
alerts, and at x1 it is 0.5 point worse because its latency term discourages the full
model even where deadlines are comfortable (the urgency weighting reduces this: 0.1312
at x1 on seed 1 with `urgency=0.2, gamma=10`). With at most 1.4 points between
`risk-split` and the fluid bound, and `ps-greedy-s0.1` already within 0.3 points of the
bound, joint optimisation has nothing left to win here. Its decision time (20 ms of
annealing plus building) fits the 100 ms epoch but is ten times that of the greedy rule.
Whether joint placement pays off must be shown on a scenario with more headroom.

## Next

- A harder, realistic scenario (feature extraction as gateway CPU work, a model ladder,
  background load), screened with the fluid bound before any method is compared on it.
- Faster QUBO construction (it now calls the processor-sharing model per option).
