"""Decision-focused option values for the epoch model.

The option value of ``ps-anneal-learned`` is ``crit x share x p(o, a) - w x fa(o)``, with
p the detection probability for the estimated attack flows a and fa the expected false
alerts; the two weights are the same for every task. Here both become functions of the
task, ``s(t) x crit x share x p(o, a) - lam(t) x fa(o)`` with
``s = exp(theta_s . phi(t))`` and ``lam = exp(theta_l . phi(t))``, and theta is trained on
decision quality (SPO+ loss with the annealer as the solver, ``scripts/train_dfl.py``)
rather than on value accuracy. With theta_s = 0 and theta_l = (log w, 0, ...) the values
are those of ``ps-anneal-learned``.
"""
import json
import math

import numpy as np

from .learned import LearnedPSAnneal, task_features

PHI = ['bias', 'flood', 'prefilter_score', 'log_flows', 'log_packets', 'criticality', 'deadline_s',
       'share_10', 'share_20', 'share_40', 'z_10', 'z_20', 'z_40', 'log_alerts_40',
       'log_expected_false_40', 'log_attack']


def phi_raw(f, a_hat):
    """Task features of the value weights (unstandardised; ``bias`` is 1)."""
    return [1.0, 1.0 if f['stream'] == 'flood' else 0.0, f['prefilter_score'], math.log1p(f['n_flows']),
            math.log1p(f['packets']), f['criticality'], f['deadline_s'], f['share_10'], f['share_20'],
            f['share_40'], min(max(f['z_10'], -5.0), 20.0), min(max(f['z_20'], -5.0), 20.0),
            min(max(f['z_40'], -5.0), 20.0), math.log1p(f['alerts_40']), math.log1p(f['expected_false_40']),
            math.log1p(max(a_hat, 0.0))]


def standardise(raw, mean, std):
    x = (np.asarray(raw, float) - mean) / std
    x[..., 0] = 1.0
    return x


class DFLPSAnneal(LearnedPSAnneal):
    """ps-anneal-learned with task-dependent value weights trained on decision quality."""
    name = 'ps-anneal-dfl'

    def __init__(self, model_path, params_path, **kw):
        super().__init__(model_path, **kw)
        with open(params_path, encoding='utf-8') as fh:
            p = json.load(fh)
        if p['phi'] != PHI:
            raise ValueError(f'{params_path}: other features')
        self.mean, self.std = np.array(p['mean']), np.array(p['std'])
        self.theta_s, self.theta_l = np.array(p['theta_s']), np.array(p['theta_l'])
        self.weights = {}

    def decide(self, engine, now, tasks):
        # Features before the base class updates its evidence: the same moment as when
        # the training data were recorded.
        for ev in {id(e): e for e in self.horizons.values()}.values():
            ev.update(engine, now)
        feats = [task_features(t, self.horizons) for t in tasks]
        out_ = None
        if tasks:
            from threadpoolctl import threadpool_limits
            from .learned import feature_row
            with threadpool_limits(1):
                a_hat = self.model.predict([feature_row(f) for f in feats])
            x = standardise([phi_raw(f, a) for f, a in zip(feats, a_hat)], self.mean, self.std)
            s, lam = np.exp(np.clip(x @ self.theta_s, -20, 20)), np.exp(np.clip(x @ self.theta_l, -30, 10))
            for t, si, li in zip(tasks, s, lam):
                self.weights[t.id] = (float(si), float(li))
        out_ = super().decide(engine, now, tasks)
        return out_

    def value(self, t, o, share=1.0):
        if o.node is None:
            return 0.0
        s, lam = self.weights[t.id]
        a = self.attack_estimate(t)
        p = 1.0 - (1.0 - o.f * self.recall[o.model]) ** a
        fa = o.f * t.n_flows * (1.0 - t.prefilter_score) * self.sc.model(o.model).false_positive_rate
        return s * share * t.criticality * p - lam * fa
