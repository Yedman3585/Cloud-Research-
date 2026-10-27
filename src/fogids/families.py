"""Scenario families for the learning experiments (docs/learning-plan.md).

Each family changes one assumption of the alert-based value formula. A family is a
function of an x3 smokescreen configuration whose ``seed`` is already set: everything it
draws (device profiles, campaigns, surges, topologies) comes from its own generator
seeded by the configuration seed, so every seed is a different instance of the family
and the traffic stream of the base generator is left alone.

- F1 ``f1-detectors``: device profiles per gateway scale the models' false-positive rate
  and recall; a quarter of the gateways change profile at 30 s.
- F2 ``f2-botnet``: the low-rate attacks become staged campaigns (scan, brute force,
  Mirai flood) that spread to gateways of the same fog subtree.
- F3 ``f3-evasive``: low-rate attackers keep their rate under about two alerts per 10 s
  on a fully analysed gateway and move to fresh gateways every 8-12 s. (Planned as
  reacting to their own alerts; implemented on a fixed schedule, since the traffic of a
  run is generated before it starts.)
- F4 ``f4-surges``: benign flash crowds, x5-x10 on groups of gateways for 5-15 s, a third
  of the extra flows taken for heavy hitters by the header pass.
- F5 ``f5-infra``: other trees (fan-out), fog capacity, uplink bandwidth and load.
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
X1 = ROOT / 'configs/ids-place-smokescreen.json'
FLOOD = ('ddos', 'dos', 'mirai')
LOW_RATE = (('recon', 40.0), ('web', 15.0), ('bruteforce', 15.0))


def _rng(c, family):
    return random.Random(1_000_003 * int(c['seed']) + family)


def _gateways(c):
    t = c['topology']
    return t['fog_nodes'] * t['gateways_per_fog']


def _floods(c):
    return [a for a in c['attacks'] if a['class'] in FLOOD]


def f1_detectors(c):
    rng = _rng(c, 1)
    n = _gateways(c)
    types = {'camera': {'fpr': 3.0, 'recall': 0.6}, 'hub': {'fpr': 1.5, 'recall': 0.85},
             'sensor': {'fpr': 0.3, 'recall': 1.0}}
    names = list(types)
    initial = [rng.choices(names, weights=(0.3, 0.3, 0.4))[0] for _ in range(n)]
    changes = []
    for g in rng.sample(range(n), n // 4):  # firmware update or device swap at 30 s
        changes.append((30.0, g, rng.choice([x for x in names if x != initial[g]])))
    c['profiles'] = {'types': types, 'initial': initial, 'changes': changes}
    c['name'] = 'smokescreen-f1-detectors'


def f2_botnet(c):
    rng = _rng(c, 2)
    t = c['topology']
    n, per_fog = _gateways(c), t['gateways_per_fog']
    horizon = float(c['horizon_s'])
    infected = {}
    queue = [(rng.uniform(3.0, 12.0), g) for g in rng.sample(range(n), 3)]
    while queue:
        queue.sort()
        t_inf, g = queue.pop(0)
        if g in infected or t_inf > horizon - 15.0:
            continue
        infected[g] = t_inf
        fog = g // per_fog
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.8:
                target = fog * per_fog + rng.randrange(per_fog)
            else:
                target = rng.randrange(n)
            if target not in infected:
                queue.append((t_inf + rng.uniform(5.0, 15.0), target))
    bursts = []
    for g, t_inf in sorted(infected.items()):
        stages = (('recon', t_inf, 10.0, 40.0), ('bruteforce', t_inf + 5.0, 15.0, 15.0),
                  ('mirai', t_inf + 15.0, horizon - t_inf - 15.0, 1500.0))
        for cls, start, dur, rate in stages:
            dur = min(dur, horizon - start)
            if dur > 0.5:
                bursts.append({'class': cls, 'start_s': round(start, 3), 'duration_s': round(dur, 3),
                               'ramp_s': 0.5, 'gateways': [g], 'flows_per_s': rate})
    c['attacks'] = _floods(c) + bursts
    c['name'] = 'smokescreen-f2-botnet'


def f3_evasive(c):
    rng = _rng(c, 3)
    n = _gateways(c)
    bursts = []
    for cls, _ in LOW_RATE:
        t, recent = 5.0, []
        while t < 57.0:
            dur = min(rng.uniform(8.0, 12.0), 57.0 - t)
            fresh = [g for g in range(n) if g not in recent]
            gws = sorted(rng.sample(fresh, 3))
            recent = (recent + gws)[-6:]
            bursts.append({'class': cls, 'start_s': round(t, 3), 'duration_s': round(dur, 3), 'ramp_s': 0.0,
                           'gateways': gws, 'flows_per_s': 0.25})
            t += dur
    c['attacks'] = _floods(c) + bursts
    c['name'] = 'smokescreen-f3-evasive'


def f4_surges(c):
    rng = _rng(c, 4)
    n = _gateways(c)
    c['surges'] = [{'gateways': sorted(rng.sample(range(n), 4)), 'start_s': round(rng.uniform(5.0, 50.0), 3),
                    'duration_s': round(rng.uniform(5.0, 15.0), 3), 'ramp_s': 1.0,
                    'factor': round(rng.uniform(5.0, 10.0), 3), 'flood_share': 1 / 3}
                   for _ in range(8)]
    c['name'] = 'smokescreen-f4-surges'


def f5_infra(c):
    rng = _rng(c, 5)
    fogs, per_fog = rng.choice([(2, 8), (4, 4), (4, 8), (8, 4), (4, 16), (8, 8)])
    t = c['topology']
    t['fog_nodes'], t['gateways_per_fog'] = fogs, per_fog
    t['fog'] = {**t['fog'], 'mips': round(t['fog']['mips'] * rng.uniform(0.5, 2.0))}
    t['edge'] = {**t['edge'], 'uplink_bytes_s': round(t['edge']['uplink_bytes_s'] * rng.uniform(0.25, 1.0))}
    load = rng.choice([1, 2, 3, 4, 5])
    x1 = json.loads(X1.read_text(encoding='utf-8'))
    c['traffic']['benign_flows_per_s'] = x1['traffic']['benign_flows_per_s'] * load
    rates = {a['class']: a['flows_per_s'] for a in x1['attacks']}
    n = fogs * per_fog
    for a in c['attacks']:
        if a['class'] in ('ddos', 'dos'):
            a['flows_per_s'] = round(rates[a['class']] * load, 1)
        # The attack pattern is laid out for 32 gateways: fold it onto fewer, repeat it
        # on every block of 32 when there are more.
        if n <= 32:
            a['gateways'] = sorted({g % n for g in a['gateways']})
        else:
            a['gateways'] = sorted(g + 32 * b for g in a['gateways'] for b in range(n // 32))
    c['name'] = f'smokescreen-f5-infra-{fogs}x{per_fog}-x{load}'


FAMILIES = {
    'f1-detectors': f1_detectors,
    'f2-botnet': f2_botnet,
    'f3-evasive': f3_evasive,
    'f4-surges': f4_surges,
    'f5-infra': f5_infra,
}
