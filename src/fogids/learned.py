"""Learned value estimate for the epoch model.

The option value of ps-anneal-alerts depends on one unknown per task: the number of
attack flows it contains (``attack_estimate``). Here that number is predicted by a
regression model from features visible at decision time (pre-filter score, stream, size,
the gateway's alert evidence over 10, 20 and 40 s); everything else (value formula,
annealer, budget, repair) is unchanged. The features are the ones recorded by
``scripts/collect_values.py``; models are trained by ``scripts/train_values.py``.
"""
import pickle

from .smokescreen_qubo import AlertEvidence, AlertPSAnneal

HORIZONS = (('10', 10.0), ('20', 20.0), ('40', 40.0))
FEATURES = ['flood', 'prefilter_score', 'n_flows', 'packets', 'criticality', 'deadline_s'] + [
    f'{k}_{h}' for h, _ in HORIZONS for k in ('alerts', 'expected_false', 'analysed_recall', 'share', 'z')]
PROFILES = ('camera', 'hub', 'sensor')
# Second feature set (scenario families): bytes per packet, the gateway's device profile
# and the alert evidence of the other gateways of its fog subtree over 40 s.
FEATURES_V2 = FEATURES + ['bytes_per_packet'] + [f'device_{p}' for p in PROFILES] + [
    f'nb_{k}_40' for k in ('alerts', 'expected_false', 'analysed_recall', 'share')]


def feature_row(f, names=FEATURES):
    """Feature vector from a recorded task row (``collect_values.py``) or ``task_features``."""
    return [1.0 if f['stream'] == 'flood' else 0.0] + [float(f[k]) for k in names[1:]]


def task_features(t, evidence, sc=None):
    """The recorded features of task ``t``; ``evidence`` maps '10', '20', '40' to an
    up-to-date ``AlertEvidence`` with that horizon. With the scenario ``sc`` also the
    features of ``FEATURES_V2``."""
    f = {'stream': t.stream, 'prefilter_score': t.prefilter_score, 'n_flows': t.n_flows,
         'packets': t.packets, 'criticality': t.criticality, 'deadline_s': t.deadline_s}
    for h, _ in HORIZONS:
        ev = evidence[h]
        A, E, N = ev.acc.get(t.gateway, (0.0, 0.0, 0.0))
        f.update({f'alerts_{h}': A, f'expected_false_{h}': E, f'analysed_recall_{h}': N,
                  f'share_{h}': ev.share(t.gateway), f'z_{h}': ev.z(t.gateway)})
    if sc is not None:
        f['bytes_per_packet'] = t.raw_bytes / max(1, t.packets)
        profile = sc.device_profile(t.gateway, t.release_s)  # device types are known to the operator
        f.update({f'device_{p}': 1.0 if profile == p else 0.0 for p in PROFILES})
        ev = evidence['40']
        parent = sc.node(t.gateway).parent
        A = E = N = 0.0
        for gw, (a, e, n) in ev.acc.items():
            if gw != t.gateway and sc.node(gw).parent == parent:
                A, E, N = A + a, E + e, N + n
        f.update({'nb_alerts_40': A, 'nb_expected_false_40': E, 'nb_analysed_recall_40': N,
                  'nb_share_40': max(0.0, A - E) / N if N else 0.0})
    return f


def load_model(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


class LearnedPSAnneal(AlertPSAnneal):
    """ps-anneal-alerts with the attack-flow estimate predicted by a trained model."""
    name = 'ps-anneal-learned'

    def __init__(self, model_path, floor=3.0, **kw):
        super().__init__(floor=floor, **kw)
        bundle = load_model(model_path)
        if bundle['features'] not in (FEATURES, FEATURES_V2):
            raise ValueError(f'{model_path}: trained on other features')
        self.model = bundle['model']
        self.features = bundle['features']
        self.predicted = {}

    def reset(self, sc):
        super().reset(sc)
        self.horizons = {h: (self.evidence if s == self.horizon_s else AlertEvidence(sc, s))
                         for h, s in HORIZONS}

    def attack_estimate(self, t):
        return max(self.floor, self.predicted[t.id])

    def decide(self, engine, now, tasks):
        for ev in {id(e): e for e in self.horizons.values()}.values():
            ev.update(engine, now)
        if tasks:
            from threadpoolctl import threadpool_limits
            sc = self.sc if self.features == FEATURES_V2 else None
            X = [feature_row(task_features(t, self.horizons, sc), self.features) for t in tasks]
            with threadpool_limits(1):  # parallel workers would fight over OpenMP threads
                predicted = self.model.predict(X)
            for t, a in zip(tasks, predicted):
                self.predicted[t.id] = float(a)
        return super().decide(engine, now, tasks)


class TrueAttackPSAnneal(AlertPSAnneal):
    """Reference: the same value formula with the true attack flows of each task, as
    defined by the training target (low-rate flows in a residual task, attack flows in a
    flood task). Separates the error of the estimate from the form of the value."""
    name = 'ps-anneal-true-a'

    def attack_estimate(self, t):
        from .smokescreen import FLOOD
        if t.stream == 'flood':
            return float(sum(a for _, a in t.attacks))
        return max(self.floor, float(sum(a for c, a in t.attacks if c not in FLOOD)))
