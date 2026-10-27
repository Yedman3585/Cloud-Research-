"""Adaptive alert formula: the competitor of the learned estimates (learning-plan.md, L1+L2).

Same structure as ``ps-anneal-alerts`` (attack share of a gateway = alerts in excess of
the expected false alerts, over the analysed recall), with two parts updated online:

- **L1, false-positive rate.** The expected false alerts use the calibrated rate times a
  factor rho per group of gateways (the device profile when it is known, otherwise all
  gateways). rho is a low quantile (20 %) over the group's gateways of alerts / expected
  false alerts in the current window: attacked gateways only raise the ratio, so the low
  quantile follows the false-positive level of the group. rho also scales the false-alert
  term of the option value.
- **L2, change detection.** A two-sided Page-Hinkley test on each gateway's excess-alert
  share per analysed task; when it fires, the gateway's window is cut to the last 5 s, so
  that old evidence neither hides a new attack nor keeps a finished one alive.

Recall is not adapted (the labels it needs arrive late and the false-positive rate is the
larger effect in the families).
"""
from collections import deque

import numpy as np

from .smokescreen_qubo import AlertEvidence, AlertPSAnneal


class AdaptiveEvidence(AlertEvidence):
    def __init__(self, sc, horizon_s=40.0, quantile=0.2, min_expected=20.0, ph_delta=0.01, ph_lambda=0.3,
                 keep_s=5.0):
        super().__init__(sc, horizon_s)
        self.quantile, self.min_expected = quantile, min_expected
        self.ph_delta, self.ph_lambda, self.keep_s = ph_delta, ph_lambda, keep_s
        self.rho_group = {}
        self.changes = 0

    def _group(self, gateway, now):
        p = self.sc.device_profile(gateway, now)
        return 'all' if p is None else p

    def update(self, engine, now):
        if not hasattr(self, 'acc') or getattr(self, '_engine', None) is not engine:
            self._engine, self._seen, self.acc, self._window, self._ph = engine, 0, {}, {}, {}
        alerts = engine.alerts
        while self._seen < len(alerts) and alerts[self._seen].time_s <= now:
            a = alerts[self._seen]
            self._seen += 1
            if a.stream != 'residual':
                continue
            item = (a.time_s, a.quiet_alerts, a.analysed_flows * self.fpr[a.model],
                    a.analysed_flows * self.recall[a.model])
            self._window.setdefault(a.gateway, deque()).append(item)
            s = self.acc.setdefault(a.gateway, [0.0, 0.0, 0.0])
            for k in range(3):
                s[k] += item[k + 1]
            self._page_hinkley(a.gateway, item, now)
        for gw, q in self._window.items():
            s = self.acc[gw]
            while q and q[0][0] < now - self.horizon_s:
                item = q.popleft()
                for k in range(3):
                    s[k] -= item[k + 1]
        ratios = {}
        for gw, (A, E, _) in self.acc.items():
            if E >= self.min_expected:
                ratios.setdefault(self._group(gw, now), []).append(A / E)
        for g, r in ratios.items():
            self.rho_group[g] = float(np.quantile(r, self.quantile))

    def _page_hinkley(self, gw, item, now):
        _, A, E, N = item
        if N <= 0:
            return
        x = (A - self.rho(gw) * E) / N
        st = self._ph.setdefault(gw, {'n': 0, 'mean': 0.0, 'up': 0.0, 'up_min': 0.0, 'down': 0.0, 'down_max': 0.0})
        st['n'] += 1
        st['mean'] += (x - st['mean']) / st['n']
        st['up'] += x - st['mean'] - self.ph_delta
        st['up_min'] = min(st['up_min'], st['up'])
        st['down'] += x - st['mean'] + self.ph_delta
        st['down_max'] = max(st['down_max'], st['down'])
        if st['up'] - st['up_min'] > self.ph_lambda or st['down_max'] - st['down'] > self.ph_lambda:
            self.changes += 1
            q, s = self._window[gw], self.acc[gw]
            while q and q[0][0] < now - self.keep_s:
                old = q.popleft()
                for k in range(3):
                    s[k] -= old[k + 1]
            self._ph[gw] = {'n': 0, 'mean': 0.0, 'up': 0.0, 'up_min': 0.0, 'down': 0.0, 'down_max': 0.0}

    def rho(self, gateway):
        now = getattr(self._engine, 't', 0.0) if hasattr(self, '_engine') else 0.0
        return self.rho_group.get(self._group(gateway, now), 1.0)

    def share(self, gateway):
        A, E, N = self.acc.get(gateway, (0.0, 0.0, 0.0))
        return max(0.0, A - self.rho(gateway) * E) / N if N else 0.0

    def z(self, gateway):
        A, E, _ = self.acc.get(gateway, (0.0, 0.0, 0.0))
        E = self.rho(gateway) * E
        return (A - E) / (E + 1.0) ** 0.5


class AdaptiveAlertPSAnneal(AlertPSAnneal):
    """ps-anneal-alerts with the adaptive evidence (L1 + L2)."""
    name = 'ps-anneal-adaptive'

    def __init__(self, floor=3.0, quantile=0.2, ph_delta=0.01, ph_lambda=0.3, **kw):
        super().__init__(floor=floor, **kw)
        self.ev_kw = {'quantile': quantile, 'ph_delta': ph_delta, 'ph_lambda': ph_lambda}

    def reset(self, sc):
        super().reset(sc)
        self.evidence = AdaptiveEvidence(sc, self.horizon_s, **self.ev_kw)

    def value(self, t, o, share=1.0):
        if o.node is None:
            return 0.0
        a = self.attack_estimate(t)
        p = 1.0 - (1.0 - o.f * self.recall[o.model]) ** a
        fa = o.f * t.n_flows * (1.0 - t.prefilter_score) * self.sc.model(o.model).false_positive_rate
        return share * t.criticality * p - self.fa_weight * self.evidence.rho(t.gateway) * fa
