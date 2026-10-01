# IDS-PLACE v1: Problem Specification

IDS-PLACE is the scheduling problem studied in this project: **risk-aware online
placement of deep-learning IDS inference in an edge-fog-cloud tree under attack-driven
traffic bursts.** This document fixes the problem before any solver is built, so that
methods are compared on a frozen benchmark rather than the problem being tuned to a
method. Code: [`problem.py`](../src/fogids/problem.py) (model and instance format) and
[`generator.py`](../src/fogids/generator.py) (seeded instances).

Status: **v1 draft (milestones M1 and M2).** Model costs and recalls in the shipped configurations
are uncalibrated placeholders until milestone M4 (measurement on CICIoT2023).

## 1. System

- **Topology.** A tree with one cloud root, F fog nodes, and E edge gateways (G per fog).
  Node m has speed `mips_m`, memory, idle/busy power, a cloud price per 1000 MI, and an
  uplink to its parent with bandwidth `bw_m` (bytes/s) and latency `lat_m` (s).
- **Pipeline.** IoT traffic at gateway g is cut into feature windows of length W. Feature
  extraction runs locally at the gateway. **IDS inference is the only placeable stage.**
  Alerts are delivered to the cloud and are outside the latency measured in v1.
- **Models.** A set K of IDS model variants (v1: `light`, `full`). Variant k costs
  `fixed_mi_k + mi_per_flow_k * flows` MI, needs `memory_k` MB, takes `load_k` s to cold-start
  on a node where it is not resident, and detects attack class c with recall `recall_k[c]`.

## 2. Tasks

A task i is one feature window (or a chunk of at most `max_flows_per_task` flows):

| Field | Meaning | Visible to schedulers |
|---|---|---|
| `release_s` | features ready at the gateway (absolute time) | yes |
| `gateway` | originating edge node | yes |
| `n_flows`, `input_bytes` | size of the window | yes |
| `deadline_s` | relative deadline from its risk class | yes |
| `prefilter_score` | noisy attack score in [0, 1] from a cheap pre-filter | yes |
| `criticality` | weight of the devices behind the gateway | yes |
| `attack_flows`, `label` | ground truth (dominant class) | **no**, evaluation only |

Risk weight: `w_i = criticality_i * (0.1 + prefilter_score_i)`. Deadline class: short
(`high_s`) if `prefilter_score_i >= high_risk_threshold`, otherwise long (`low_s`).

## 3. Decisions

Time is divided into epochs of length Δ (v1: 0.1 s). At the end of each epoch the
scheduler sees the tasks released during it and the system state, and chooses for each
task one eligible node and one model:

    y[i, m, k] = 1  if task i runs on node m with model k

Eligibility in v1 is the gateway and its ancestors (`gateway -> fog -> cloud`), which is
what the simulator's tree supports. Lateral fog-to-fog offloading needs links that the
tree does not have and is deferred. Within a node, tasks share the CPU; how it is shared
(time sharing, or weighted shares from a NUM layer) is part of the scheduler design.

Constraints: each task gets exactly one (node, model) pair; a model must fit in the
node's memory (loading a non-resident model costs `load_k` and may evict others).

## 4. Objective and Metrics

The scheduler minimizes, per epoch, a weighted sum of

1. risk-weighted detection latency `Σ w_i T_i`, where `T_i` = release-to-inference-completion
   time including uplink queueing and CPU sharing;
2. expected missed detection `Σ w_i (1 - recall_k[class_i])` (estimated from the pre-filter
   score, since the true class is hidden);
3. energy of edge and fog nodes, and cloud cost.

Evaluation reports (per risk class): p50/p95/p99 time-to-detect, deadline-miss rate,
weighted missed detection (using ground truth), energy, cloud cost, and scheduler
decision time. Decision time is charged to the simulation clock.

## 5. Instance Generator

Traffic per gateway and window: benign flows ~ Poisson(rate · W · J) with lognormal
jitter J; each attack burst adds Poisson(flows_per_s · overlap) flows, where the overlap
integrates a linear ramp-up. Gateway windows are staggered by seeded phases. Attack
bursts target a seeded subset of gateways. The pre-filter score is the attack fraction
plus Gaussian noise, clipped to [0, 1].

Reproducibility: all randomness uses only `random.Random.random()` (stable across
Python versions) and stored floats are rounded to 9 digits. A unit test pins the SHA-256
of the small instance; it passes on Python 3.10–3.14.

Shipped configurations (placeholders, uncalibrated):

| Config | Fog × gateways | Horizon | Bursts | Tasks | Peak tasks/epoch | Full-model load on edge+fog |
|---|---|---|---|---|---|---|
| `ids-place-small` | 2 × 4 | 30 s | DDoS, Recon | 350 | 10 | 0.83 |
| `ids-place-medium` | 4 × 8 | 60 s | Mirai, DDoS, Recon, Web | 2700 | 21 | 1.04 |
| `ids-place-large` | 8 × 16 | 60 s | Mirai, DDoS, Recon, Web | 10783 | 69 | 1.24 |

A load above 1 means edge and fog cannot run every window with the full model, so model
choice and cloud offloading matter. Generate and inspect an instance:

```bash
fogids generate --config configs/ids-place-small.json      # writes artifacts/instances/
fogids generate --config configs/ids-place-small.json --seed 7
fogids describe artifacts/instances/ids-place-small-s11.json
```

## 6. Online Simulation (milestone M2)

`org.fogids.IdsOnline` builds the instance tree as iFogSim `FogDevice` nodes with one IDS
module per node, schedules every task release, and wakes at every epoch boundary. When
tasks were released since the previous epoch, it writes one JSON line to stdout and
blocks on stdin; simulator logging is redirected to stderr.

```text
-> {"type":"epoch","t":1.0,"epoch":10,"tasks":[ids],"completed":[[id,finish_s],...],
    "nodes":{"gw0":{"outstanding":2,"resident":["light"]},...}}
<- {"assignments":[[id,"fog0","full"],...],"delay_s":0.0}
-> {"type":"end","t":...,"results":[{"id","node","model","dispatch_s","finish_s","cold_start"},...]}
```

Rules enforced by the driver: every released task is assigned exactly once, to its
gateway or an ancestor, with a known model. Assignments take effect after `delay_s`.
A task released exactly on an epoch boundary belongs to that epoch. A cold start is
modelled as `load_s` seconds of extra CPU work on the target node, with
least-recently-used residency within node memory. Inference work and uplink transfers
use iFogSim's FogDevice queues and CloudSim time sharing, as validated by the analytic
tests in `tests/test_online.py`.

Python side: [`bridge.py`](../src/fogids/bridge.py) runs the driver and a policy;
[`policies.py`](../src/fogids/policies.py) holds the reference policies;
[`metrics.py`](../src/fogids/metrics.py) computes per-class deadline misses, latency
percentiles over completed tasks (with the unfinished count), expected missed attack
flows (ground truth used only here), edge and fog energy from executed work under the
linear power model, and cloud cost. Runs end when all tasks finish or 30 s after the
last release; unfinished tasks are misses.

First results with uncalibrated placeholder costs (seed as in the shipped configs):

| Instance | Policy | Miss rate (all) | Miss rate (short deadline) | p95 latency, s | Unfinished |
|---|---|---|---|---|---|
| small | edge-light | 0.431 | 0.981 | 2.29 | 0 |
| small | cloud-full | 0.429 | 0.974 | 2.98 | 0 |
| small | greedy-finish | 0.346 | 0.786 | 0.66 | 0 |
| medium | cloud-full | 0.875 | 0.998 | 63.0 | 899 |
| medium | greedy-finish | 0.247 | 0.601 | 0.64 | 0 |

Short-deadline tasks are mostly missed under every baseline: during bursts, a 0.3 s
deadline is shorter than the edge uplink transfer of large windows. Whether this is a
property of the problem or of the placeholder parameters is decided by calibration (M4).

## 7. Open Points for v1 Freeze

- Whether a scheduler may re-plan tasks that were placed but have not started (this
  enlarges the per-epoch decision and is relevant for QUBO size).
- Model eviction policy when memory is full.
- Calibrated costs and recalls (M4); trace-driven arrivals from CICIoT2023 in addition to
  the synthetic bursts.
- Short-deadline feasibility under bursts (see section 6) after calibration.
