# IDS-PLACE: Online Placement of Deep-Learning IDS Inference

A research prototype for scheduling IDS inference tasks across edge, fog and cloud
nodes. A QUBO-based annealer places tasks every 100 ms epoch; iFogSim2 executes
the assignments with network queues and CPU sharing. Gradient-boosted trees learn
the QUBO value coefficients from IDS alerts.

## Results Summary

On the smokescreen scenario (DDoS cover for low-rate attacks, 32 gateways, x3 load):

| Policy | Objective | Description |
|---|---|---|
| greedy-v2 | 0.386 | Expert rule baseline |
| SA-TOCA (tuned offline with labels) | 0.252 | Best static split from literature |
| ps-anneal-alerts (ours, formula) | 0.255 | Online, no labels needed |
| ps-anneal-learned (ours, GBT) | 0.248 | One setting, beats SA-TOCA |

Across 10 conditions including 5 scenario families (F1–F5), the learned estimate has
the lowest mean objective (0.285) and the smallest worst-case shortfall (5.0 pp)
among all non-oracle estimates.

Full results: [`docs/ids-place-full-documentation.md`](docs/ids-place-full-documentation.md).

## Quick Start

```bash
python scripts/bootstrap.py
source .venv/bin/activate
fogids doctor
```

### Run the smokescreen scenario (Python simulator)

```bash
python scripts/robustness.py --variants x3 --policies greedy-v2 ps-anneal-alerts ps-anneal-learned --seeds 2 3 4 5 6
```

### Run in iFogSim

```bash
python scripts/smokescreen_ifogsim.py --config configs/ids-place-smokescreen-x3.json --variant x3 --policies ps-anneal-alerts ps-anneal-learned
```

### Train the learned model

```bash
python scripts/collect_values.py --variant x3 --seeds 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
python scripts/train_values.py --loo
```

### Run the IDS-PLACE v1 demo (minimal QUBO)

```bash
fogids ids-demo
```

### Run the tests

```bash
python -m unittest discover -s tests -v
```

## Architecture

Three layers connected by configuration files:

1. **ML layer** (CICIoT2023 → two IDS models) provides the detection quality
   and inference cost that the scheduler trades off.
2. **Task layer** (seeded generator) builds reproducible instances from topology,
   model parameters and traffic patterns.
3. **Simulation layer** (iFogSim2) executes scheduler decisions with real queues
   and CPU sharing, producing the ground truth.

The QUBO epoch model uses **exact processor-sharing finish times** as quadratic
constraints and **estimated option values** as linear coefficients. The value
estimate evolved through the project:

- Hand-crafted formula from IDS alert statistics (40 s window)
- Gradient-boosted trees predicting attack flows from 21 scheduler-visible features
- Decision-focused weights trained on scheduling regret

See [`docs/architecture.md`](docs/architecture.md) for the full code map.

## IDS Models

Trained on CICIoT2023 (559,800 rows, 7 attack classes + benign):

| Model | Architecture | Balanced accuracy | Benign FPR | Cost per 1000 flows |
|---|---|---|---|---|
| light | MLP 39→64→32→8 | 0.716 | 25.6% | 0.17 ms |
| full | GRU, 16-record context | 0.804 | 12.7% | 12.3 ms |

Details: [`docs/calibration.md`](docs/calibration.md).

## Competitors

- **SA-TOCA** (Sudyana et al., IEEE TSC 2025): static split tuned offline with
  ground-truth labels. Ties at x3, loses under shift.
- **FOG-QIEA** (Hammouda & Shalaby et al., Sci. Rep. 2026): quantum-inspired
  evolutionary algorithm. Cannot compete (0.44 vs 0.255).
- Oracle values (exact ground-truth task values): 0.230 at x3.

## Documentation

| File | Description |
|---|---|
| [`docs/ids-place-full-documentation.md`](docs/ids-place-full-documentation.md) | Unified documentation: idea, data, all experiments |
| [`docs/research-overview.md`](docs/research-overview.md) | Project overview in Russian |
| [`docs/smokescreen.md`](docs/smokescreen.md) | Smokescreen scenario: method and results |
| [`docs/learned-values.md`](docs/learned-values.md) | Learned coefficients: model, training, all results |
| [`docs/families.md`](docs/families.md) | Scenario families F1–F5 |
| [`docs/ids-place.md`](docs/ids-place.md) | IDS-PLACE v1 specification (milestones M1–M5) |
| [`docs/calibration.md`](docs/calibration.md) | CICIoT2023 calibration |
| [`docs/ps-model.md`](docs/ps-model.md) | Processor-sharing model |
| [`docs/learning-plan.md`](docs/learning-plan.md) | Pre-registered experimental plan |
| [`docs/architecture.md`](docs/architecture.md) | Code architecture |

## Installation

Requirements: Python 3.10+, JDK 17, access to GitHub and PyPI.

```bash
python scripts/bootstrap.py
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
fogids doctor
```

For calibration (CICIoT2023 data in `data/CICIoT2023/`):
```bash
pip install -e '.[calibration]'
python scripts/calibrate.py all --profile mac
```

## Key Files

```
src/fogids/
  smokescreen.py           Python simulator (Engine), scenario generator
  smokescreen_qubo.py      QUBO policies: ps-anneal, ps-anneal-alerts, oracle
  fast_anneal.py           Numba-compiled annealer (~120k moves / 50 ms)
  ps_model.py              Exact processor-sharing model
  learned.py               LearnedPSAnneal: GBT-based value estimation
  adaptive.py              Adaptive alert formula (online FPR, Page-Hinkley)
  dfl.py                   Decision-focused learning
  families.py              Scenario families F1–F5
  baselines.py             SA-TOCA and FOG-QIEA implementations
  bounds.py                Fluid LP lower bound
  problem.py, generator.py IDS-PLACE model and seeded instance generator
  bridge.py                Java-Python epoch bridge
  policies.py              v1 reference policies
  qubo.py, experiment.py   Minimal QUBO demo
java/org/fogids/
  IdsStaged.java           iFogSim2 driver for the smokescreen scenario
  IdsOnline.java           v1 epoch-based driver
  IdsBatch.java            Minimal demo batch driver
scripts/
  train_values.py          GBT model training (Poisson loss)
  train_dfl.py             DFL training (zeroth-order descent)
  collect_values.py        Training data collection
  robustness.py            Multi-condition evaluation
  calibrate.py             CICIoT2023 calibration
configs/                   Scenario configurations
tests/                     Unit and integration tests
docs/                      Documentation
```
