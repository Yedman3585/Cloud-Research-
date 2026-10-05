# Fog/Edge IDS Scheduling Research

A runnable research prototype for assigning IDS inference tasks to edge, fog,
and cloud nodes. Python builds a placement QUBO and solves it with classical
simulated annealing (SA); iFogSim2 executes the assignments with network queues
and CPU sharing. A greedy baseline and a small exact oracle are included.

The formulation adapts the penalty-encoding pattern from
[Q-GARS](https://arxiv.org/abs/2603.23127). It is not a reproduction of its ranking
scheduler, SQA solver, or theoretical guarantees. See the
[model and implementation notes (Russian)](docs/ids-minimal.md) for equations,
assumptions, and tests. Reviewed papers are in [papers/](papers/README.md).

## Quick Start

From the repository root, with the environment already installed:

```bash
source .venv/bin/activate
fogids doctor
fogids ids-demo
```

The default experiment uses six synthetic, ready-to-process feature windows and
three nodes. It runs greedy and QUBO + SA, each with and without measured decision
delay. Outputs are written to `artifacts/ids-small/`, including `summary.json`,
per-task CSV results, the QUBO, and Java logs. No real DL inference is executed.

To use a different configuration or output directory:

```bash
fogids ids-demo --config configs/ids-small.json --output artifacts/my-run
```

Rerunning into the same output directory overwrites files with the same names.
The tiny exact-oracle demo accepts 1–8 tasks and the fixed edge → fog → cloud topology.

### IDS-PLACE Instances

The research problem, **IDS-PLACE** (risk-aware online placement of DL-based IDS
inference under attack bursts), is specified in [docs/ids-place.md](docs/ids-place.md).
A seeded generator builds instances: an edge-fog-cloud tree, IDS model variants and a
stream of feature-window tasks with synthetic attack bursts. The base configs use
placeholder model parameters; `configs/ids-place-*-calibrated*.json` use model costs,
recalls and false-positive rates measured on CICIoT2023 at load scales x1, x3 and x10
(see [docs/calibration.md](docs/calibration.md)).

```bash
fogids generate --config configs/ids-place-small.json   # artifacts/instances/*.json
fogids describe artifacts/instances/ids-place-small-s11.json
```

Instances run online in iFogSim through an epoch-based Java-Python bridge
(`org.fogids.IdsOnline` + [`bridge.py`](src/fogids/bridge.py)). At every epoch with new
tasks, Java sends a state snapshot as a JSON line and waits while a Python policy
decides; simulated time stands still meanwhile, and decision time can be charged to
the clock (`--decision-time none|measured|<seconds>`). Four reference policies are
included: `edge-light`, `fog-full`, `cloud-full` and `greedy-finish`.

```bash
fogids simulate --config configs/ids-place-small.json   # artifacts/runs/<instance>/
fogids simulate --config configs/ids-place-medium.json --policy greedy-finish --decision-time measured
```

Each run writes the instance, per-task CSV files, simulator logs, and `summary.json`
with deadline-miss rates per risk class, latency percentiles, missed detection
(per class and macro-averaged), the false-alert rate on benign flows, energy, cloud
cost and decision time.

Run the tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The 42 tests cover QUBO expansion, feasible-assignment costs, bounded slack,
seeded SA, configuration validation, Java integration, the IDS-PLACE problem model
(eligibility, transfer and compute times, validation), and the instance generator
(determinism, a cross-platform fingerprint, burst and chunking behaviour), and the
online bridge against analytic timings (epoch boundaries, transfer plus compute,
cold starts, CPU sharing, decision delay, snapshot state, invalid assignments), the
detection metrics (false alerts, per-class and macro missed detection), and the
calibration helpers (one of them needs NumPy and is skipped without it). Simulation checks
include local compute time, cloud network transfer, shared CPU execution,
decision delay, and unfinished tasks.

## How Python Connects to iFogSim

The current bridge uses **CSV files and a JVM subprocess**. Python computes an
assignment before starting the simulation; Java reads and executes that assignment.
Java does not call Python during simulation, and there is no persistent RPC service yet.

```mermaid
flowchart LR
    C["configs/ids-small.json"] --> P["Python: greedy / QUBO + SA"]
    P --> I["nodes.csv + tasks.csv"]
    I --> J["Java: IdsBatch / iFogSim"]
    J --> O["results.csv + logs"]
    O --> R["Python: summary.json"]
```

1. [`experiment.py`](src/fogids/experiment.py) loads and validates the scenario,
   constructs a feasible greedy incumbent, and calls [`qubo.py`](src/fogids/qubo.py).
   The exact oracle enumerates small assignments for comparison; it does not
   replace the SA result.
2. Python writes node definitions and task assignments, compiles
   [`IdsBatch.java`](java/org/fogids/IdsBatch.java) with `javac`, and launches
   `org.fogids.IdsBatch` through `subprocess`. Arguments specify input/output paths,
   release time, decision delay, and simulation horizon. Base classes and libraries
   come from the pinned iFogSim checkout.
3. Java creates three `FogDevice` nodes in an edge → fog → cloud chain, with one
   IDS `AppModule` per node. Ready feature windows enter at edge as `Tuple` objects
   addressed to the selected replica. iFogSim models uplink transmission and queues;
   CloudSim models time-shared CPU execution.
4. Java records completion on `CLOUDLET_RETURN`. Python reads these results and
   computes latency and deadline misses. Each of the four modes uses a fresh JVM.

### File Contract and Units

| Output file | Contents |
|---|---|
| `config.json` | Copy of the scenario configuration |
| `nodes.csv` | `name, mips, uplink_bytes_s, uplink_latency_s` |
| `<mode>-tasks.csv` | `id, node_index, compute_mi, input_bytes, deadline_s` |
| `<mode>-results.csv` | `task_id, node, completed, latency_s, deadline_s, missed` |
| `qubo.json` | Variable labels, upper-triangular coefficients, constant offset |
| `<mode>.log` | Java process output |
| `summary.json` | Assignments, surrogate costs, optimization timings, completion metrics |

Node indices follow the configuration: 0=edge, 1=fog, 2=cloud. Time is in
seconds, work in MI, CPU speed in MIPS, sizes in bytes, and bandwidth in bytes/s.
`deadline_s` is relative to feature-window readiness. Latency ends at inference
completion; feature extraction and alert delivery are outside this initial scenario.
Priorities are supplied inputs, not ground-truth attack labels.

### Decision Time Is Part of the Experiment

| Mode | Delay before applying the assignment |
|---|---|
| `greedy` | Zero: isolates the effect of placement |
| `qubo_sa` | Zero: isolates the effect of placement |
| `greedy_with_delay` | Measured greedy computation time |
| `qubo_sa_with_delay` | Greedy + QUBO construction/encoding + SA + decoding |

For `with_delay` modes, Java injects tasks at `release + decision_delay`, while
latency and deadlines are measured from `release`. Optimization therefore consumes
simulated time. Exact-oracle evaluation, file writing, compilation, and JVM startup
are excluded from decision delay as test-harness overhead. Java process wall time
is reported separately; **it is not a measurement of future RPC latency**.

Unfinished tasks count as deadline misses. Mean and maximum latency refer only
to completed tasks. Deadline comparison allows 10 microseconds of numerical tolerance;
the CPU update interval is 0.01 seconds. SA is seeded, but measured wall-clock
runtime depends on the machine and load.

### Local CPU Adapter

The pinned upstream `TupleScheduler` inherits an empty `getCurrentRequestedMips()`.
Periodic `PowerHost` updates consequently revoked CPU allocation in this finite-batch
scenario. The local `InferenceScheduler` reports full node CPU demand while busy
and zero while idle; standard time sharing distributes it among tasks within the
replica. The upstream checkout is unchanged. Analytic integration tests validate
single-task execution and two tasks sharing one CPU.

## QUBO Scope

Binary `y[i,m]` assigns task i to node m. The model uses one-hot task assignment
and integer node work budgets encoded with bounded binary slack. Its nonnegative
surrogate combines predicted latency, weighted tardiness, and a pairwise contention
proxy. A node work budget is an admission constraint over a configured compute
horizon, not a guarantee that all assigned tasks finish within that horizon.

Penalty `P = U + 1`, where U is the feasible incumbent's surrogate cost, ensures
that a global QUBO minimum is feasible under this model's integer-residual and
nonnegative-cost assumptions. SA is approximate: decoded assignments are checked,
and the feasible incumbent remains available. Greedy may fail to construct an
incumbent for some packing instances; the prototype reports that failure explicitly.

The default instance has 27 binary variables. The exact oracle checks the same
surrogate objective, not optimality of measured simulation latency. The included
solver is dependency-free **classical SA**, not SQA or a quantum hardware call.
Equations and the distinction from Q-GARS are documented in
[docs/ids-minimal.md](docs/ids-minimal.md).

## Installation and Reproducibility

- Python: tested with **3.12.14** in an isolated `.venv` environment.
- Java: tested with **Temurin 17.0.20.1**, using the existing system Java.
- iFogSim2: official **v2.0.0** release, commit
  `643c433b9d6c9f031a2e31f129f2b2c6c7fae835`.
- Source: https://github.com/Cloudslab/iFogSim
- Version lock: [`configs/ifogsim.lock.json`](configs/ifogsim.lock.json).

After cloning, use JDK 17, Python 3.10+, and access to GitHub/PyPI:

```bash
python3.12 scripts/bootstrap.py
source .venv/bin/activate
fogids doctor
fogids ids-demo
```

On Windows (PowerShell), with JDK 17 first on `PATH`:

```powershell
python scripts\bootstrap.py
.venv\Scripts\fogids doctor
.venv\Scripts\fogids ids-demo
.venv\Scripts\python -m unittest discover -s tests -v
```

The upstream commit checked during setup,
`5f68d3947e450d8d2b4af42670be819206be68c9`, includes CloudSim 7 integration and
requires APIs unavailable on Java 17, such as `List.getLast()`. This project therefore
pins iFogSim2 v2.0.0. Compilation uses ISO-8859-1 for older upstream GUI sources;
our Java adapter is compiled separately with UTF-8. Deprecation warnings in upstream
do not prevent the build.

Bootstrap does not switch an existing checkout to another version or change
system-wide installations. Libraries come from upstream `jars/`; precompiled
upstream `out/` and `output/` are not used. Vendor sources, `.venv`, build outputs,
and generated experiment results are excluded from Git and regenerated locally.

To rebuild the upstream classes or run the original installation smoke test:

```bash
fogids build
fogids run
fogids run --class-name org.fog.test.perfeval.VRGameFog --timeout 120
```

`fogids run` launches `VRGameFog` and writes `artifacts/VRGameFog.log`.
It is separate from the IDS experiment. `ids-demo` compiles the local Java adapter
automatically and builds upstream classes if they are missing.

## Verified Runs (September 2026)

The Python-to-iFogSim bridge was verified end to end on two platforms: bootstrap
(clone of iFogSim v2.0.0 and compilation of 327 upstream sources), `fogids doctor`,
all 10 tests, and `fogids ids-demo` with the default configuration.

| Platform | Python | JDK | Bootstrap and doctor | Tests |
|---|---|---|---|---|
| Windows 11, PowerShell | 3.14.6 | Temurin 17.0.20.1 | OK | 10/10 |
| Linux (cloud workspace) | 3.11.15 | OpenJDK 21.0.10 | OK | 10/10 |

The macOS workflow above is unchanged. On Windows, JDK 17 was placed first on
`PATH` for the session because another JDK (23) was the system default:

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

Two cross-platform fixes were required: `bootstrap.py` uses `.venv\Scripts\python.exe`
on Windows, and `scripts/ifogsim.py` writes forward-slash paths to the javac argument
file, because javac treats backslashes inside quoted argfile entries as escapes.

`ids-demo` results (6 tasks, seed 7; deadline misses out of 6):

| Mode | Linux: assignment | Linux: misses | Windows: assignment | Windows: misses |
|---|---|---|---|---|
| `greedy` | [1,1,0,2,2,2] | 5 | [1,1,0,2,2,2] | 5 |
| `qubo_sa` | [2,1,1,2,0,2] | 3 | [1,2,0,2,1,2] | 1 |
| `greedy_with_delay` | same as greedy | 6 | same as greedy | 6 |
| `qubo_sa_with_delay` | same as qubo_sa | 6 | same as qubo_sa | 6 |

Observations:

- **SA is seeded but not reproducible across platforms.** With the same seed, SA returned
  surrogate 14.585 on Linux and the exact-oracle optimum 14.580 on Windows. The likely
  cause is a different Python version (3.11 vs 3.14). Future work: a version-stable RNG
  (for example NumPy PCG64) and recording interpreter and JDK versions in `summary.json`.
- **Decision time dominates.** Pure-Python SA took 0.35 s (Linux) and 0.50 s (Windows),
  which erased the placement gain: all six deadlines are missed in `qubo_sa_with_delay`.
- **The surrogate is a weak proxy for simulated outcomes.** Simulating all 48 feasible
  assignments in iFogSim gave a Spearman correlation between surrogate and deadline
  misses of 0.01 (0.13 for priority-weighted latency). An assignment with zero misses
  exists but ranks 15th of 48 by surrogate; the surrogate optimum itself misses one
  deadline. Shared uplink queues are not modelled in the surrogate.
- **Deadlines lie on the boundary.** Tasks 2 and 4 finish exactly at their deadlines, so a
  0.15 ms decision delay changes greedy from 5 to 6 misses. A single instance is therefore
  not evidence; randomized instances and multiple seeds are needed.

## Repository Structure

```text
src/fogids/               Python CLI, IDS-PLACE model, generator, online bridge, policies,
                          metrics, QUBO, SA, oracle, demo driver
java/org/fogids/IdsOnline.java  Epoch-based online IDS-PLACE driver for iFogSim
java/org/fogids/IdsBatch.java   Finite IDS inference batch (minimal demo)
configs/ids-small.json    Synthetic tasks, nodes, and SA budget (minimal demo)
configs/ids-place-*.json  IDS-PLACE generator configs: small, medium, large
configs/ifogsim.lock.json Pinned simulator version
docs/ids-minimal.md       Equations, assumptions, and implementation notes (Russian)
docs/ids-place.md         IDS-PLACE v1 problem specification
tests/                   QUBO unit tests and Java integration tests
scripts/ifogsim.py        Upstream Java build and launch wrapper
scripts/bootstrap.py     Environment restoration
papers/                  Research papers and bibliography
vendor/ifogsim/           Unmodified upstream checkout, ignored by Git
build/ifogsim/classes/    Compiled Java classes, ignored by Git
artifacts/               Generated results and logs, ignored by Git
```

## Current Limits and Next Steps

This is a synthetic, single ready batch with no background workload. It does not
yet model state becoming stale during optimization. There is no real IDS dataset,
DL execution, model selection, full feature → inference → alert DAG, ML training,
SQA/QPU backend, or Hedge guarantee. RAM is fixed; energy and cloud cost are not
optimization objectives in this version. A single example is not evidence of
scheduler superiority.

The next integration step is a persistent Java–Python protocol carrying a
`snapshot_id`, simulation timestamp, ready tasks, residual resources, assignments,
and solver runtime, with a freshness check before applying decisions. Adapting
Hedge to discrete, state-changing schedules requires separate feedback and regret
assumptions; Q-GARS guarantees do not transfer automatically.
