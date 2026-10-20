"""Record training data for a learned value estimate (Python simulator).

Runs ps-anneal-alerts on the given seeds and writes, for every epoch, the epoch model
the annealer solved: every option with its loads, budget-relevant data, the value the
policy estimated from the alerts and the exact (oracle) value from ground truth, plus
the scheduler-visible features of every task (pre-filter score, stream, size, the
gateway's alert evidence over 10, 20 and 40 s). A second file holds every task's
outcome. Seeds default to 7-26, disjoint from the evaluation seeds 2-6.

    python scripts/collect_values.py --seeds 7 8 9 --workers 2

Output: artifacts/value-data/seed-<k>.epochs.jsonl.gz and seed-<k>.tasks.jsonl.gz.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def collect(job):
    config_path, seed, out = job
    from fogids.generator import load_config
    from fogids.smokescreen import Engine, generate
    from fogids.smokescreen_qubo import AlertEvidence, AlertPSAnneal

    class Recorder(AlertPSAnneal):
        name = 'ps-anneal-alerts-recorder'

        def reset(self, sc):
            super().reset(sc)
            self.extra = {h: AlertEvidence(sc, h) for h in (10.0, 20.0)}
            self.records = []
            self.now = 0.0

        def decide(self, engine, now, tasks):
            self.now = now
            for ev in self.extra.values():
                ev.update(engine, now)
            self.evidence.update(engine, now)
            features = []
            for t in tasks:
                f = {'id': t.id, 'gateway': t.gateway, 'release_s': t.release_s, 'deadline_s': t.deadline_s,
                     'stream': t.stream, 'n_flows': t.n_flows, 'packets': t.packets,
                     'prefilter_score': t.prefilter_score, 'criticality': t.criticality}
                for h, ev in (('10', self.extra[10.0]), ('20', self.extra[20.0]), ('40', self.evidence)):
                    A, E, N = ev.acc.get(t.gateway, (0.0, 0.0, 0.0))
                    f.update({f'alerts_{h}': A, f'expected_false_{h}': E, f'analysed_recall_{h}': N,
                              f'share_{h}': ev.share(t.gateway), f'z_{h}': ev.z(t.gateway)})
                features.append(f)
            self._features = features
            out_ = super().decide(engine, now, tasks)
            return out_

        def build_ps(self, est, tasks, now):
            m = super().build_ps(est, tasks, now)
            counts, scale = self.counts, 300.0
            options = []
            for i, (t, row) in enumerate(zip(tasks, m.options)):
                for k, o in enumerate(row):
                    if o.node is None:
                        oracle = 0.0
                    else:
                        det = self.sc.detection(t, o)
                        oracle = scale * sum(det[c] / counts[c] for c, _ in t.attacks) \
                            - self.fa_weight * self.sc.false_alerts(t, o)
                    options.append({'task': i, 'node': o.node, 'model': o.model, 'f': o.f, 'site': o.site,
                                    'value': m.values[i][k], 'oracle_value': oracle,
                                    'loads': [[list(r), a] for r, a in m.stages[i][k].items()]})
            self.records.append({'t': now, 'tasks': [t.id for t in tasks], 'budgets': m.budgets,
                                 'horizons': m.horizons, 'kappa': m.kappa,
                                 'fixed': [[v, [[list(r), a] for r, a in loads.items()], B, D]
                                           for v, loads, B, D in m.fixed],
                                 'options': options})
            return m

    config = load_config(config_path)
    config['seed'] = seed
    sc = generate(config)
    policy = Recorder(floor=3.0)
    started = time.perf_counter()
    engine = Engine(sc)
    feats = {}
    original = policy.decide

    def decide(engine_, now, tasks):
        out_ = original(engine_, now, tasks)
        for f in policy._features:
            feats[f['id']] = f
        return out_

    policy.decide = decide
    r = engine.run(policy)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / f'seed-{seed}.epochs.jsonl.gz', 'wt', encoding='utf-8') as f:
        for rec in policy.records:
            f.write(json.dumps(rec) + '\n')
    with gzip.open(out / f'seed-{seed}.tasks.jsonl.gz', 'wt', encoding='utf-8') as f:
        for t in sc.tasks:
            o = engine.decisions.get(t.id)
            fin = engine.finish.get(t.id)
            row = {**feats.get(t.id, {'id': t.id}), 'attacks': dict(t.attacks), 'benign_flows': t.benign_flows,
                   'chosen': None if o is None or o.node is None else [o.node, o.model, o.f, o.site],
                   'finish_s': fin, 'on_time': fin is not None and fin <= t.release_s + t.deadline_s + 1e-6}
            f.write(json.dumps(row) + '\n')
    return seed, r['objective'], time.perf_counter() - started, len(policy.records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ids-place-smokescreen-x3.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(7, 27)))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/value-data')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    seeds = [s for s in args.seeds
             if not (args.resume and (args.output / f'seed-{s}.tasks.jsonl.gz').exists())]
    print(f'{len(args.seeds) - len(seeds)} seeds kept, {len(seeds)} to do', flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(collect, (str(args.config), s, args.output)) for s in seeds]
        for future in as_completed(futures):
            seed, objective, seconds, epochs = future.result()
            print(f'seed {seed}: objective {objective:.4f}, {epochs} epochs ({seconds:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
