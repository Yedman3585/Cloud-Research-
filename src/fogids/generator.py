"""Seeded IDS-PLACE instance generator.

Traffic model (per edge gateway, per feature window of length W):

* benign flows ~ Poisson(rate_g * W * J), with a per-window lognormal jitter J;
* each active attack burst adds Poisson(flows_per_s * overlap) flows, where the
  overlap is the time the burst (with a linear ramp-up) covers the window;
* gateway windows are staggered by seeded phases (``stagger_windows``);
* the window becomes ready after feature extraction at the gateway and is cut
  into tasks of at most ``max_flows_per_task`` flows, attack flows spread
  proportionally;
* a noisy pre-filter score (attack fraction + Gaussian noise) is the only
  attack signal schedulers see; it also selects the deadline class.

All randomness is drawn from ``random.Random.random()``, the one method whose
output Python guarantees to be identical across versions, and stored floats are
rounded, so a seed produces the same instance on every platform.
"""
import json
import math
from pathlib import Path
import random

from .problem import BENIGN, Instance, Model, Node, Task

GENERATOR_VERSION = 1


class Stream:
    """Version-stable sampling built only on uniform draws."""

    def __init__(self, seed):
        self._rng = random.Random(seed)

    def uniform(self):
        return self._rng.random()

    def normal(self):
        u1 = max(self.uniform(), 1e-300)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * self.uniform())

    def poisson(self, mean):
        if mean <= 0:
            return 0
        if mean < 30:  # Knuth multiplication method
            limit, k, p = math.exp(-mean), 0, self.uniform()
            while p > limit:
                k += 1
                p *= self.uniform()
            return k
        return max(0, int(math.floor(mean + math.sqrt(mean) * self.normal() + 0.5)))

    def lognormal_unit_mean(self, sigma):
        return math.exp(sigma * self.normal() - 0.5 * sigma * sigma) if sigma > 0 else 1.0

    def permutation(self, n):
        order = list(range(n))
        for i in range(n - 1, 0, -1):  # Fisher-Yates
            j = int(self.uniform() * (i + 1))
            order[i], order[j] = order[j], order[i]
        return order


def _pick(stream, count, selector):
    """Indices chosen by an explicit list or by a fraction of a seeded permutation."""
    if isinstance(selector, list):
        if any(not 0 <= i < count for i in selector):
            raise ValueError('Gateway index out of range')
        return sorted(selector)
    if not 0 <= selector <= 1:
        raise ValueError('Gateway fraction must lie in [0, 1]')
    return sorted(stream.permutation(count)[:math.ceil(selector * count)])


def _burst_overlap(burst, t0, t1):
    """Attack intensity integrated over [t0, t1], in seconds at full rate."""
    start, end, ramp = burst['start_s'], burst['start_s'] + burst['duration_s'], burst.get('ramp_s', 0.0)

    def mass(t):  # integral of the ramped indicator from start to t
        t = min(max(t, start), end)
        if ramp <= 0 or t >= start + ramp:
            return (t - start) - (ramp / 2 if ramp > 0 else 0.0)
        return (t - start) ** 2 / (2 * ramp)
    return max(0.0, mass(t1) - mass(t0))


def build_topology(c, stream):
    topo = c['topology']
    def spec(tier):
        return {**topo[tier], 'resident': tuple(topo[tier].get('resident', ()))}
    cloud = Node(name='cloud', tier='cloud', parent=None, uplink_bytes_s=None, uplink_latency_s=None,
                 **spec('cloud'))
    fogs = [Node(name=f'fog{f}', tier='fog', parent='cloud', **spec('fog')) for f in range(topo['fog_nodes'])]
    n_gateways = topo['fog_nodes'] * topo['gateways_per_fog']
    critical = set(_pick(stream, n_gateways, c['risk']['critical_gateways']))
    gateways = [Node(name=f'gw{g}', tier='edge', parent=f'fog{g // topo["gateways_per_fog"]}',
                     criticality=c['risk']['critical_weight'] if g in critical else 1.0, **spec('edge'))
                for g in range(n_gateways)]
    return [cloud, *fogs, *gateways]


def generate(c):
    """Build and validate an instance from a generator configuration dictionary."""
    stream = Stream(c['seed'])
    nodes = build_topology(c, stream)
    gateways = [n for n in nodes if n.tier == 'edge']
    models = [Model(name=m['name'], fixed_mi=m['fixed_mi'], mi_per_flow=m['mi_per_flow'],
                    memory_mb=m['memory_mb'], load_s=m['load_s'], recall=dict(m['recall']))
              for m in c['models']]
    traffic, risk, deadlines = c['traffic'], c['risk'], c['deadlines']
    bursts = [{**b, 'targets': set(_pick(stream, len(gateways), b['gateways']))} for b in c['attacks']]
    rates = traffic['benign_flows_per_s']
    W, horizon = c['window_s'], c['horizon_s']
    # Gateways close their windows at independent, seeded phases rather than in lockstep.
    phases = [stream.uniform() * W if traffic.get('stagger_windows', True) else 0.0 for _ in gateways]
    raw = []
    for w in range(int(round(horizon / W))):
        for g, gw in enumerate(gateways):
            t0, t1 = phases[g] + w * W, phases[g] + (w + 1) * W
            if t1 > horizon:
                continue
            rate = rates[g % len(rates)] if isinstance(rates, list) else rates
            benign = stream.poisson(rate * W * stream.lognormal_unit_mean(traffic['benign_jitter']))
            attacks = {}
            for b in bursts:
                if g in b['targets']:
                    k = stream.poisson(b['flows_per_s'] * _burst_overlap(b, t0, t1))
                    if k:
                        attacks[b['class']] = attacks.get(b['class'], 0) + k
            total = benign + sum(attacks.values())
            if total == 0:
                continue
            attack_total = sum(attacks.values())
            label = max(sorted(attacks), key=lambda a: attacks[a]) if attacks else BENIGN
            ready = t1 + traffic['feature_s_per_flow'] * total
            chunks = math.ceil(total / traffic['max_flows_per_task'])
            done = 0
            for k in range(chunks):
                n = total // chunks + (1 if k < total % chunks else 0)
                # Attack flows spread proportionally; cumulative rounding keeps the exact total.
                a = attack_total * (done + n) // total - attack_total * done // total
                done += n
                score = min(1.0, max(0.0, a / n + risk['prefilter_noise'] * stream.normal()))
                high = score >= deadlines['high_risk_threshold']
                raw.append((ready, g, Task(
                    id=0, gateway=gw.name, release_s=round(ready, 9),
                    deadline_s=deadlines['high_s'] if high else deadlines['low_s'],
                    n_flows=n, attack_flows=a, label=label if a else BENIGN,
                    input_bytes=traffic['header_bytes'] + traffic['bytes_per_flow'] * n,
                    prefilter_score=round(score, 9), criticality=gw.criticality)))
    raw.sort(key=lambda r: (r[0], r[1]))
    tasks = [Task(**{**t.__dict__, 'id': i}) for i, (_, _, t) in enumerate(raw)]
    meta = {'generator_version': GENERATOR_VERSION, 'config_name': c['name'],
            'calibrated': c.get('calibrated', False)}
    return Instance(name=c['name'], seed=c['seed'], horizon_s=float(horizon), window_s=float(W),
                    epoch_s=float(c['epoch_s']), nodes=nodes, models=models, tasks=tasks, meta=meta).validate()


def load_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
