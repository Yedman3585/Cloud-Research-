"""Scenario v2: a flood as a smokescreen for low-rate attacks.

Measured on CICIoT2023-style traffic (``docs/calibration.md``), feature extraction
costs 7-8 us per packet on an M2, far more than inference, and a DDoS burst offers
a Raspberry-Pi-class gateway about eleven times the packets it can parse. An IDS
under a flood must therefore decide, besides where to run which model, **what share
of the traffic to analyse at all**. Attackers use exactly this: a flood that
saturates the monitor while a low-rate attack (reconnaissance, brute force, web
exploits) slips through.

Mechanisms added to IDS-PLACE (everything else as in ``problem.py``):

* every window of a gateway is split by a cheap header pass into a *flood* stream
  (heavy-hitter flows) and a *residual* stream (the rest); the split is imperfect
  (``separation_error``). Each stream is cut into tasks;
* a task has packets as well as flows; feature extraction is CPU work of
  ``extract_mi_per_packet`` per analysed packet plus ``hash_mi_per_packet`` per packet
  for the header pass. It runs at the gateway (then features are shipped), or the
  gateway ships sampled raw packets and the chosen node extracts (``site='node'``);
* a decision is (node, model, fraction f, extraction site); f is the share of
  flows analysed (flow sampling); f = 0 drops the task;
* detection is per task and attack class: a task analysed at f with a model of
  per-flow recall r detects class c with probability 1 - (1 - f r_c)^(a_c) for its
  a_c flows of c, if it finishes by its deadline (0 otherwise). Analysed benign
  flows are flagged at the model's false-positive rate. Per-flow quality does not
  depend on f (measured, ``scripts/sampling_eval.py``).

Objective (lower is better): macro-averaged missed detection over attack classes
(per task containing the class), plus ``fa_weight`` times false alerts per benign
flow.

The module provides the generator, an exact event-driven simulator of the resources
(processor-sharing CPUs, FIFO uplinks, as in iFogSim, ``ps_model.py``), a
policy-side estimator, reference policies and an offline fluid lower bound.
"""
from dataclasses import dataclass
import heapq
import math
import random

from .generator import Stream, _burst_overlap, _pick, build_topology
from .problem import Instance, Model
from .ps_model import ps_finish_times

FLOOD = ('ddos', 'dos', 'mirai')
TOL = 1e-6


@dataclass(frozen=True)
class STask:
    id: int
    gateway: str
    release_s: float
    deadline_s: float
    stream: str  # 'flood' or 'residual' (scheduler-visible)
    n_flows: int
    attacks: tuple  # ((class, flows), ...) -- ground truth, never visible to schedulers
    packets: int
    raw_bytes: int  # bytes of all packets of the task
    feature_bytes: int  # bytes of the extracted features of all flows
    header_bytes: int
    prefilter_score: float
    criticality: float

    @property
    def attack_flows(self):
        return sum(a for _, a in self.attacks)

    @property
    def benign_flows(self):
        return self.n_flows - self.attack_flows

    @property
    def weight(self):
        return self.criticality * (0.1 + self.prefilter_score)

    @property
    def estimated_attack_flows(self):
        return self.n_flows * self.prefilter_score


@dataclass(frozen=True)
class Option:
    node: str | None  # None: dropped
    model: str | None
    f: float
    site: str  # 'gw': extract at the gateway; 'node': ship raw packets, extract at node


DROP = Option(None, None, 0.0, 'gw')


@dataclass(frozen=True)
class Alert:
    """Alerts raised by one analysed task (scheduler-visible feedback)."""
    time_s: float
    gateway: str
    stream: str
    model: str
    f: float
    analysed_flows: int
    flood_alerts: int
    quiet_alerts: int


class Scenario:
    def __init__(self, config, topology, tasks, params):
        self.config, self.topology, self.tasks, self.p = config, topology, tasks, params
        self.epoch_s, self.horizon_s = config['epoch_s'], config['horizon_s']
        self.models = topology.models
        self.fractions = tuple(params['fractions'])
        self.classes = sorted({c for t in tasks for c, _ in t.attacks})

    def node(self, name):
        return self.topology.node(name)

    def model(self, name):
        return self.topology.model(name)

    def eligible(self, task):
        return self.topology.ancestors(task.gateway)

    def links(self, task, node):
        return self.topology.path_links(task.gateway, node)

    def options(self, task):
        out = [DROP]
        for f in self.fractions:
            for node in self.eligible(task):
                for m in self.models:
                    out.append(Option(node, m.name, f, 'gw'))
                    if node != task.gateway:
                        out.append(Option(node, m.name, f, 'node'))
        return out

    # Costs of an option ---------------------------------------------------
    def stages(self, task, opt):
        """[(kind, resource, amount)] in order: ('cpu', node, MI) or ('link', link, bytes)."""
        if opt.node is None:
            return []
        p = self.p
        flows = max(1, math.ceil(opt.f * task.n_flows))
        infer = self.model(opt.model).work_mi(flows)
        hashing = task.packets * p['hash_mi_per_packet']
        extract = opt.f * task.packets * p['extract_mi_per_packet']
        links = self.links(task, opt.node)
        if opt.site == 'gw':
            out = [('cpu', task.gateway, hashing + extract)]
            if opt.node == task.gateway:
                return out + [('cpu', task.gateway, infer)]
            size = task.header_bytes + opt.f * task.feature_bytes
            return out + [('link', l, size) for l in links] + [('cpu', opt.node, infer)]
        size = task.header_bytes + opt.f * task.raw_bytes
        return ([('cpu', task.gateway, hashing)] + [('link', l, size) for l in links]
                + [('cpu', opt.node, extract + infer)])

    def standalone_s(self, task, opt):
        """Latency of the option on idle resources (no queueing)."""
        t = 0.0
        for kind, res, amount in self.stages(task, opt):
            spec = self.node(res)
            t += amount / spec.mips if kind == 'cpu' else amount / spec.uplink_bytes_s + spec.uplink_latency_s
        return t

    # Outcomes -------------------------------------------------------------
    def detection(self, task, opt, on_time=True):
        """{class: probability of detecting the class in this task}."""
        if opt.node is None or not on_time:
            return {c: 0.0 for c, _ in task.attacks}
        m = self.model(opt.model)
        return {c: 1.0 - (1.0 - opt.f * m.recall.get(c, 0.0)) ** a for c, a in task.attacks}

    def false_alerts(self, task, opt, on_time=True):
        if opt.node is None or not on_time:
            return 0.0
        return opt.f * task.benign_flows * self.model(opt.model).false_positive_rate

    def class_counts(self):
        counts = {}
        for t in self.tasks:
            for c, _ in t.attacks:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def cost(self, task, opt, counts=None, on_time=True):
        """Contribution of one task to the objective (exact, uses ground truth)."""
        counts = counts or self.class_counts()
        det = self.detection(task, opt, on_time)
        miss = sum((1.0 - det[c]) / counts[c] for c, _ in task.attacks) / max(1, len(counts))
        benign = sum(t.benign_flows for t in self.tasks)
        return miss + self.p['fa_weight'] * self.false_alerts(task, opt, on_time) / max(1, benign)

    def evaluate(self, decisions, finish):
        """Metrics of a run: decisions {id: Option}, finish {id: time or None}."""
        counts = self.class_counts()
        det = {c: 0.0 for c in counts}
        fa, benign, dropped, late, analysed = 0.0, 0, 0, 0, 0
        for t in self.tasks:
            opt = decisions.get(t.id, DROP)
            on_time = (opt.node is not None and finish.get(t.id) is not None
                       and finish[t.id] <= t.release_s + t.deadline_s + TOL)
            if opt.node is None:
                dropped += 1
            elif not on_time:
                late += 1
            for c, p in self.detection(t, opt, on_time).items():
                det[c] += p
            fa += self.false_alerts(t, opt, on_time)
            benign += t.benign_flows
            analysed += t.benign_flows * opt.f if on_time else 0
        detected = {c: det[c] / counts[c] for c in counts}
        quiet = [c for c in counts if c not in FLOOD]
        flood = [c for c in counts if c in FLOOD]
        missed = 1.0 - sum(detected.values()) / max(1, len(detected))
        return {
            'objective': missed + self.p['fa_weight'] * fa / max(1, benign),
            'missed_macro': missed,
            'missed_quiet': 1.0 - sum(detected[c] for c in quiet) / max(1, len(quiet)),
            'missed_flood': 1.0 - sum(detected[c] for c in flood) / max(1, len(flood)),
            'detected_by_class': detected,
            'false_alert_rate': fa / max(1, benign),
            'benign_coverage': analysed / max(1, benign),
            'dropped': dropped / max(1, len(self.tasks)),
            'late': late / max(1, len(self.tasks)),
        }


# Generator -------------------------------------------------------------------

def generate(c):
    """Scenario from a configuration (``configs/ids-place-smokescreen*.json``)."""
    stream = Stream(c['seed'])
    nodes = build_topology(c, stream)
    models = [Model(name=m['name'], fixed_mi=m['fixed_mi'], mi_per_flow=m['mi_per_flow'],
                    memory_mb=m['memory_mb'], load_s=m['load_s'], recall=dict(m['recall']),
                    false_positive_rate=m.get('false_positive_rate', 0.0)) for m in c['models']]
    topology = Instance(name=c['name'], seed=c['seed'], horizon_s=float(c['horizon_s']),
                        window_s=float(c['window_s']), epoch_s=float(c['epoch_s']),
                        nodes=nodes, models=models, tasks=[]).validate()
    gateways = [n for n in nodes if n.tier == 'edge']
    v2, traffic, risk, deadlines = c['v2'], c['traffic'], c['risk'], c['deadlines']
    ref = v2['ref_mips']
    params = {
        'extract_mi_per_packet': v2['extract_us_per_packet'] * 1e-6 * ref,
        'hash_mi_per_packet': v2['hash_us_per_packet'] * 1e-6 * ref,
        'fractions': v2['fractions'], 'fa_weight': v2['fa_weight'],
    }
    ppf, pbytes = v2['packets_per_flow'], v2['packet_bytes']
    bursts = [{**b, 'targets': set(_pick(stream, len(gateways), b['gateways']))} for b in c['attacks']]
    W, horizon = c['window_s'], c['horizon_s']
    phases = [stream.uniform() * W for _ in gateways]
    eps = v2['separation_error']
    raw = []
    for w in range(int(round(horizon / W))):
        for g, gw in enumerate(gateways):
            t0, t1 = phases[g] + w * W, phases[g] + (w + 1) * W
            if t1 > horizon:
                continue
            benign = stream.poisson(traffic['benign_flows_per_s'] * W * stream.lognormal_unit_mean(traffic['benign_jitter']))
            attacks = {}
            for b in bursts:
                if g in b['targets']:
                    k = stream.poisson(b['flows_per_s'] * _burst_overlap(b, t0, t1))
                    if k:
                        attacks[b['class']] = attacks.get(b['class'], 0) + k
            # Header pass: heavy-hitter flows to the flood stream, with errors both ways.
            flood = {c_: a for c_, a in attacks.items() if c_ in FLOOD}
            quiet = {c_: a for c_, a in attacks.items() if c_ not in FLOOD}
            streams = {'flood': {'benign': 0, **{}}, 'residual': {'benign': benign}}
            for c_, a in flood.items():
                leak = min(a, stream.poisson(eps * a))
                streams['flood'][c_] = a - leak
                streams['residual'][c_] = streams['residual'].get(c_, 0) + leak
            if flood:
                leak = min(benign, stream.poisson(eps * benign))
                streams['flood']['benign'] = leak
                streams['residual']['benign'] = benign - leak
            for c_, a in quiet.items():
                streams['residual'][c_] = a
            for name in ('flood', 'residual'):
                comp = {k: v for k, v in streams[name].items() if v > 0}
                total = sum(comp.values())
                if total == 0:
                    continue
                chunks = math.ceil(total / traffic['max_flows_per_task'])
                done = {k: 0 for k in comp}
                for k in range(chunks):
                    share = {key: v * (k + 1) // chunks - v * k // chunks for key, v in comp.items()}
                    n = sum(share.values())
                    if n == 0:
                        continue
                    atk = tuple(sorted((key, v) for key, v in share.items() if key != 'benign' and v > 0))
                    a = sum(v for _, v in atk)
                    packets = sum(v * ppf.get(key, ppf['default']) for key, v in share.items())
                    raw_b = sum(v * ppf.get(key, ppf['default']) * pbytes.get(key, pbytes['default'])
                                for key, v in share.items())
                    score = min(1.0, max(0.0, a / n + risk['prefilter_noise'] * stream.normal()))
                    high = score >= deadlines['high_risk_threshold']
                    raw.append((t1, g, STask(
                        id=0, gateway=gw.name, release_s=round(t1, 9),
                        deadline_s=deadlines['high_s'] if high else deadlines['low_s'], stream=name,
                        n_flows=n, attacks=atk, packets=packets, raw_bytes=raw_b,
                        feature_bytes=traffic['bytes_per_flow'] * n, header_bytes=traffic['header_bytes'],
                        prefilter_score=round(score, 9), criticality=gw.criticality)))
                    for key in done:
                        done[key] += share[key]
    raw.sort(key=lambda r: (r[0], r[1]))
    tasks = [STask(**{**t.__dict__, 'id': i}) for i, (_, _, t) in enumerate(raw)]
    return Scenario(c, topology, tasks, params)


# Exact simulator -------------------------------------------------------------

class Engine:
    """Processor-sharing CPUs and FIFO uplinks; decisions at epoch boundaries for the
    tasks released during the previous epoch."""

    def __init__(self, sc):
        self.sc = sc
        self.t = 0.0
        self.active = {n.name: {} for n in sc.topology.nodes}  # node -> {job: remaining MI}
        self.link_free = {n.name: 0.0 for n in sc.topology.nodes}
        self.events = []  # (time, seq, job, node, work)
        self.seq = 0
        self.jobs = {}  # job -> (task, remaining stages)
        self.loaded = {n.name: set(n.resident) for n in sc.topology.nodes}
        self.finish = {}
        self.decisions = {}
        # Alert feedback: what the IDS reported, visible to the scheduler once raised.
        self.alerts = []  # Alert records in order of time
        self.emit_alerts = True
        self._rng = random.Random(sc.config['seed'] * 7919 + 17)

    def _binomial(self, n, p):
        n = int(n)
        if n <= 0 or p <= 0:
            return 0
        if n < 50:
            return sum(1 for _ in range(n) if self._rng.random() < p)
        mean, sd = n * p, math.sqrt(n * p * (1 - p))
        return max(0, min(n, int(round(mean + sd * self._rng.gauss(0.0, 1.0)))))

    def _alert(self, task, t):
        if not self.emit_alerts:
            return
        a = self.sample_alert(task, self.decisions.get(task.id), t)
        if a is not None:
            self.alerts.append(a)

    def sample_alert(self, task, opt, t):
        """Sample the alerts of an analysed task. Attack flows are reported in their
        group (flood or low-rate) with the model's recall; every false positive on a
        benign flow is reported as a low-rate attack (measured: all benign
        misclassifications of both models fall in the low-rate classes,
        ``artifacts/calibration/alert-groups.json``)."""
        if opt is None or opt.node is None or t > task.release_s + task.deadline_s + TOL:
            return None
        m = self.sc.model(opt.model)
        flood = quiet = 0
        for c, a in task.attacks:
            k = self._binomial(a, opt.f)
            hits = self._binomial(k, m.recall.get(c, 0.0))
            if c in FLOOD:
                flood += hits
            else:
                quiet += hits
        benign = self._binomial(task.benign_flows, opt.f)
        quiet += self._binomial(benign, m.false_positive_rate)
        analysed = round(opt.f * task.n_flows)
        return Alert(t, task.gateway, task.stream, opt.model, opt.f, analysed, flood, quiet)

    def _start(self, task, stages, now):
        """Run the stages from ``now``: links are reserved at once (FIFO), CPU stages
        become jobs."""
        t = now
        while stages and stages[0][0] == 'link':
            _, link, size = stages.pop(0)
            spec = self.sc.node(link)
            start = max(t, self.link_free[link])
            self.link_free[link] = start + size / spec.uplink_bytes_s
            t = self.link_free[link] + spec.uplink_latency_s
        if not stages:
            self.finish[task.id] = t
            self._alert(task, t)
            return
        _, node, work = stages.pop(0)
        job = (task.id, len(stages))
        self.jobs[job] = (task, stages)
        heapq.heappush(self.events, (t, self.seq, job, node, work))
        self.seq += 1

    def _advance(self, until):
        while True:
            t_cpu, node_done = math.inf, None
            for node, jobs in self.active.items():
                if jobs:
                    mips = self.sc.node(node).mips
                    dt = min(jobs.values()) * len(jobs) / mips
                    if self.t + dt < t_cpu:
                        t_cpu, node_done = self.t + dt, node
            t_arr = self.events[0][0] if self.events else math.inf
            t_next = min(t_cpu, t_arr, until)
            if t_next > self.t:
                for node, jobs in self.active.items():
                    if jobs:
                        served = (t_next - self.t) * self.sc.node(node).mips / len(jobs)
                        for j in jobs:
                            jobs[j] -= served
                self.t = t_next
            done_any = False
            for node, jobs in self.active.items():
                eps = 1e-9 * max(1.0, self.sc.node(node).mips)
                for j in [j for j, r in jobs.items() if r <= eps]:
                    del jobs[j]
                    task, stages = self.jobs.pop(j)
                    self._start(task, stages, self.t)
                    done_any = True
            while self.events and self.events[0][0] <= self.t + 1e-12:
                _, _, job, node, work = heapq.heappop(self.events)
                self.active[node][job] = max(work, 1e-12)
                done_any = True
            if self.t >= until and not done_any:
                return
            if t_next == math.inf:
                return

    def apply(self, now, batch, chosen):
        """Record and start the decisions of an epoch; returns {task id: stages} as
        executed (with the cold-start work of a model's first use on a node)."""
        executed = {}
        for t in batch:
            opt = chosen.get(t.id, DROP)
            self.decisions[t.id] = opt
            stages = self.sc.stages(t, opt)
            if stages and opt.model not in self.loaded[opt.node]:
                self.loaded[opt.node].add(opt.model)
                kind, node, work = stages[-1]
                stages[-1] = (kind, node, work + self.sc.model(opt.model).load_s * self.sc.node(node).mips)
            executed[t.id] = list(stages)
            if stages:
                self._start(t, list(stages), now)
        return executed

    def run(self, policy):
        sc = self.sc
        policy.reset(sc)
        tasks = sc.tasks
        k, epoch = 0, 1
        end = max(t.release_s + t.deadline_s for t in tasks) + sc.epoch_s
        while True:
            now = epoch * sc.epoch_s
            self._advance(now)
            batch = []
            while k < len(tasks) and tasks[k].release_s <= now + 1e-12:
                batch.append(tasks[k])
                k += 1
            if batch:
                self.apply(now, batch, policy.decide(self, now, batch))
            epoch += 1
            if k >= len(tasks) and now > end:
                self._advance(math.inf)
                break
        return sc.evaluate(self.decisions, self.finish)


# Policy-side estimator ---------------------------------------------------------

class Estimator:
    """Processor-sharing prediction from the engine state at an epoch (no knowledge
    of later tasks). Jobs in later stages are predicted from the current ones."""

    def __init__(self, engine, now):
        self.e, self.now, self.sc = engine, now, engine.sc
        self.active = {n: dict(j) for n, j in engine.active.items()}
        self.pending = {n: [] for n in engine.active}
        self.link_free = dict(engine.link_free)
        for t, _, job, node, work in engine.events:
            self.pending[node].append((t, job, work))
        self.deadline = {}
        for job, (task, _) in engine.jobs.items():
            self.deadline[job] = task.release_s + task.deadline_s
        # Later stages of running jobs: predicted arrival after the current stage.
        self.followups = []
        for node in list(self.active):
            fin = self.finishes(node)
            for job, (task, stages) in engine.jobs.items():
                if job in fin and stages:
                    self._chain(task, list(stages), fin[job], job)

    def _chain(self, task, stages, t, key):
        while stages and stages[0][0] == 'link':
            _, link, size = stages.pop(0)
            spec = self.sc.node(link)
            start = max(t, self.link_free[link])
            self.link_free[link] = start + size / spec.uplink_bytes_s
            t = self.link_free[link] + spec.uplink_latency_s
        if stages:
            _, node, work = stages[0]
            k = ('next', key)
            self.pending[node].append((t, k, work))
            self.deadline[k] = task.release_s + task.deadline_s

    def copy(self):
        other = Estimator.__new__(Estimator)
        other.e, other.now, other.sc = self.e, self.now, self.sc
        other.active = self.active  # never modified after construction
        other.pending = {n: list(v) for n, v in self.pending.items()}
        other.link_free, other.deadline = dict(self.link_free), dict(self.deadline)
        return other

    def finishes(self, node, extra=()):
        arrivals = list(self.pending[node]) + list(extra)
        return ps_finish_times(self.sc.node(node).mips, self.now, self.active[node], arrivals)

    def estimate(self, task, opt):
        """(predicted finish or None, number of committed tasks pushed past their deadline)."""
        stages = self.sc.stages(task, opt)
        if not stages:
            return None, 0
        t, broken, link_free = self.now, 0, dict(self.link_free)
        for i, (kind, res, amount) in enumerate(stages):
            if kind == 'link':
                spec = self.sc.node(res)
                start = max(t, link_free[res])
                link_free[res] = start + amount / spec.uplink_bytes_s
                t = link_free[res] + spec.uplink_latency_s
                continue
            if opt.model not in self.e.loaded[opt.node] and i == len(stages) - 1:
                amount += self.sc.model(opt.model).load_s * self.sc.node(res).mips
            key = ('new', task.id, i)
            before = self.finishes(res)
            after = self.finishes(res, [(t, key, amount)])
            t = after.pop(key)
            broken += sum(1 for k, f in after.items()
                          if k in self.deadline and before.get(k, f) <= self.deadline[k] + TOL < f)
        return t, broken

    def commit(self, task, opt):
        stages = self.sc.stages(task, opt)
        t = self.now
        for i, (kind, res, amount) in enumerate(stages):
            if kind == 'link':
                spec = self.sc.node(res)
                start = max(t, self.link_free[res])
                self.link_free[res] = start + amount / spec.uplink_bytes_s
                t = self.link_free[res] + spec.uplink_latency_s
                continue
            key = ('new', task.id, i)
            self.pending[res].append((t, key, amount))
            self.deadline[key] = task.release_s + task.deadline_s
            t = self.finishes(res)[key]
        return t


# Reference policies ------------------------------------------------------------

class SPolicy:
    name = 'policy'

    def reset(self, sc):
        self.sc = sc
        self.quality = sorted((m.name for m in sc.models),
                              key=lambda n: -(sum(sc.model(n).recall.values()) / len(sc.model(n).recall)
                                              - sc.model(n).false_positive_rate))
        self.fastest = min(sc.models, key=lambda m: m.mi_per_flow).name
        self.recall = {m.name: sum(m.recall.values()) / len(m.recall) for m in sc.models}

    def meets(self, est, task, opt, slack=0.0):
        own, broken = est.estimate(task, opt)
        budget = task.deadline_s * (1.0 if opt.model == self.fastest else 1.0 - slack)
        return own is not None and own <= task.release_s + budget + TOL and broken == 0, own


class NoShed(SPolicy):
    """Analyse everything (f = 1), features at the gateway; risk-split placement:
    short deadlines run the fastest model at the gateway, others the best model on the
    node predicted to finish first."""
    name = 'no-shed'

    def decide(self, engine, now, tasks):
        est, out = Estimator(engine, now), {}
        for t in sorted(tasks, key=lambda t: (-t.weight, t.id)):
            if t.deadline_s <= self.sc.config['deadlines']['high_s']:
                opt = Option(t.gateway, self.fastest, 1.0, 'gw')
            else:
                cands = [Option(n, self.quality[0], 1.0, 'gw') for n in self.sc.eligible(t)]
                opt = min(cands, key=lambda o: est.estimate(t, o)[0])
            est.commit(t, opt)
            out[t.id] = opt
        return out


class UniformShed(SPolicy):
    """Classic load shedding: per gateway and epoch, one sampling fraction for all its
    tasks, the largest one with which every task is predicted to meet its deadline;
    risk-split placement."""
    name = 'uniform-shed'

    def decide(self, engine, now, tasks):
        est, out = Estimator(engine, now), {}
        by_gw = {}
        for t in tasks:
            by_gw.setdefault(t.gateway, []).append(t)
        high = self.sc.config['deadlines']['high_s']
        for gw, group in by_gw.items():
            group = sorted(group, key=lambda t: (-t.weight, t.id))
            for f in self.sc.fractions + (0.0,):
                trial, ok = est.copy(), True
                plan = {}
                for t in group:
                    if f == 0.0:
                        plan[t.id] = DROP
                        continue
                    if t.deadline_s <= high:
                        opt = Option(gw, self.fastest, f, 'gw')
                    else:
                        cands = [Option(n, self.quality[0], f, 'gw') for n in self.sc.eligible(t)]
                        opt = min(cands, key=lambda o: trial.estimate(t, o)[0])
                    good, _ = self.meets(trial, t, opt)
                    ok &= good
                    trial.commit(t, opt)
                    plan[t.id] = opt
                if ok or f == 0.0:
                    est = trial
                    out.update(plan)
                    break
        return out


class GreedyV2(SPolicy):
    """ps-greedy extended to sampling: highest risk weight first; the largest fraction,
    then the best model, at the node and extraction site finishing first, among the
    options predicted to meet the deadline (10 % slack for models other than the
    fastest) without pushing a committed task past its deadline; else drop."""
    name = 'greedy-v2'

    def __init__(self, slack=0.1):
        self.slack = slack

    def decide(self, engine, now, tasks):
        est, out = Estimator(engine, now), {}
        for t in sorted(tasks, key=lambda t: (-t.weight, t.id)):
            chosen = DROP
            for f in self.sc.fractions:
                for model in self.quality:
                    best = None
                    for o in self.sc.options(t):
                        if o.f != f or o.model != model:
                            continue
                        good, own = self.meets(est, t, o, self.slack)
                        if good and (best is None or own < best[0]):
                            best = (own, o)
                    if best:
                        chosen = best[1]
                        break
                if chosen is not DROP:
                    break
            if chosen.node is not None:
                est.commit(t, chosen)
            out[t.id] = chosen
        return out


class StreamAware(GreedyV2):
    """Expert rule against smokescreens: flood-stream tasks are sampled at the smallest
    fraction with the fastest model at the gateway; residual tasks as in greedy-v2."""
    name = 'stream-aware'

    def decide(self, engine, now, tasks):
        est, out = Estimator(engine, now), {}
        flood = [t for t in tasks if t.stream == 'flood']
        for t in flood:
            opt = Option(t.gateway, self.fastest, min(self.sc.fractions), 'gw')
            est.commit(t, opt)
            out[t.id] = opt
        rest = [t for t in tasks if t.stream != 'flood']
        for t in sorted(rest, key=lambda t: (-t.weight, t.id)):
            chosen = DROP
            for f in self.sc.fractions:
                for model in self.quality:
                    best = None
                    for o in self.sc.options(t):
                        if o.f == f and o.model == model:
                            good, own = self.meets(est, t, o, self.slack)
                            if good and (best is None or own < best[0]):
                                best = (own, o)
                    if best:
                        chosen = best[1]
                        break
                if chosen is not DROP:
                    break
            if chosen.node is not None:
                est.commit(t, chosen)
            out[t.id] = chosen
        return out


class StreamAwareForecast(StreamAware):
    """stream-aware with a persistence forecast: before placing the residual tasks of a
    gateway, the gateway's next window is assumed to repeat this one (flood tasks at the
    smallest fraction, residual tasks at the fraction chosen last time) and is added as
    phantom work one window later, so residual sampling leaves room for it."""
    name = 'stream-aware-forecast'

    def reset(self, sc):
        super().reset(sc)
        self.last_f = {}

    def decide(self, engine, now, tasks):
        est, out = Estimator(engine, now), {}
        W = self.sc.config['window_s']
        for t in [t for t in tasks if t.stream == 'flood']:
            opt = Option(t.gateway, self.fastest, min(self.sc.fractions), 'gw')
            est.commit(t, opt)
            out[t.id] = opt
            for kind, res, amount in self.sc.stages(t, opt):
                if kind == 'cpu':
                    est.pending[res].append((now + W, ('phantom', t.id), amount))
        rest = sorted([t for t in tasks if t.stream != 'flood'], key=lambda t: (-t.weight, t.id))
        for t in rest:
            f_prev = self.last_f.get(t.gateway, 1.0)
            ghost = Option(t.gateway, self.fastest, f_prev, 'gw')
            for kind, res, amount in self.sc.stages(t, ghost):
                if kind == 'cpu' and res == t.gateway:
                    est.pending[res].append((now + W, ('phantom-r', t.id), amount))
        for t in rest:
            chosen = DROP
            for f in self.sc.fractions:
                for model in self.quality:
                    best = None
                    for o in self.sc.options(t):
                        if o.f == f and o.model == model:
                            good, own = self.meets(est, t, o, self.slack)
                            if good and (best is None or own < best[0]):
                                best = (own, o)
                    if best:
                        chosen = best[1]
                        break
                if chosen is not DROP:
                    break
            if chosen.node is not None:
                est.commit(t, chosen)
            out[t.id] = chosen
            self.last_f[t.gateway] = chosen.f if chosen.node is not None else min(self.sc.fractions)
        return out


class ValueGreedy(SPolicy):
    """Myopic value maximiser: highest risk weight first; among options predicted to
    meet the deadline without breaking committed tasks, the one with the best
    estimated detection value per unit of gateway CPU it takes (estimated attack flows
    from the pre-filter score), ties to the larger fraction."""
    name = 'value-greedy'

    def __init__(self, slack=0.1):
        self.slack = slack

    def value(self, t, o):
        if o.node is None:
            return 0.0
        a = t.estimated_attack_flows
        r = self.recall[o.model]
        fa = o.f * t.n_flows * (1 - t.prefilter_score) * self.sc.model(o.model).false_positive_rate
        return t.criticality * (1.0 - (1.0 - o.f * r) ** a) - 1e-4 * fa

    def decide(self, engine, now, tasks):
        est, out = Estimator(engine, now), {}
        for t in sorted(tasks, key=lambda t: (-t.weight, t.id)):
            best = (0.0, 0.0, DROP)
            for o in self.sc.options(t)[1:]:
                good, _ = self.meets(est, t, o, self.slack)
                if not good:
                    continue
                gw_cpu = sum(a for k, r, a in self.sc.stages(t, o) if k == 'cpu' and r == t.gateway)
                v = self.value(t, o)
                key = (v - 1e-6 * gw_cpu, o.f)
                if key > best[:2]:
                    best = (key[0], key[1], o)
            if best[2].node is not None:
                est.commit(t, best[2])
            out[t.id] = best[2]
        return out


POLICIES = {p.name: p for p in (NoShed, UniformShed, GreedyV2, StreamAware, StreamAwareForecast, ValueGreedy)}


# Fluid lower bound -------------------------------------------------------------

def fluid_bound(sc, lengths=(0.3, 0.5, 1.0, 2.0, 4.0), time_limit_s=900, scope='fog'):
    """Clairvoyant LP relaxation: fractional choice among all options of every task,
    capacity of every CPU and uplink over checked intervals for the work that must be
    served entirely inside them (as in ``bounds.fluid_bound``).

    ``scope='fog'``: solved separately for every fog subtree (its gateways, their
    uplinks, the fog node and its uplink); the cloud CPU, shared by all subtrees, is
    not constrained. ``scope='gateway'``: solved per gateway with only the gateway's
    CPU and uplink constrained (fog and cloud unlimited), for instances too large for
    the former. Dropping constraints keeps the result a valid lower bound."""
    counts = sc.class_counts()
    groups = {}
    for i, t in enumerate(sc.tasks):
        groups.setdefault(sc.node(t.gateway).parent if scope == 'fog' else t.gateway, []).append(i)
    det = {c: 0.0 for c in counts}
    total = {'objective': 0.0, 'fa': 0.0, 'options': 0, 'constraints': 0}
    for members in groups.values():
        _solve_group(sc, members, counts, lengths, time_limit_s, det, total, scope)
    detected = {c: det[c] / counts[c] for c in counts}
    benign = sum(t.benign_flows for t in sc.tasks)
    quiet = [c for c in counts if c not in FLOOD]
    flood = [c for c in counts if c in FLOOD]
    return {
        'objective': total['objective'],
        'missed_macro': 1.0 - sum(detected.values()) / max(1, len(detected)),
        'missed_quiet': 1.0 - sum(detected[c] for c in quiet) / max(1, len(quiet)),
        'missed_flood': 1.0 - sum(detected[c] for c in flood) / max(1, len(flood)),
        'detected_by_class': detected,
        'false_alert_rate': total['fa'] / max(1, benign),
        'options': total['options'], 'constraints': total['constraints'],
    }


def _solve_group(sc, task_ids, counts, lengths, time_limit_s, det, total, scope):
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    from .bounds import _intervals

    options, cost = [], []
    members = {}  # (kind, resource) -> [(window start, window end, amount, column)]
    for row, i in enumerate(task_ids):
        t = sc.tasks[i]
        for o in sc.options(t):
            if o.node is not None and sc.standalone_s(t, o) > t.deadline_s + TOL:
                continue
            stages = sc.stages(t, o)
            col = len(options)
            options.append((row, i, o))
            cost.append(sc.cost(t, o, counts))
            durations = []
            for kind, res, amount in stages:
                spec = sc.node(res)
                durations.append(amount / spec.mips if kind == 'cpu'
                                 else amount / spec.uplink_bytes_s + spec.uplink_latency_s)
            for s, (kind, res, amount) in enumerate(stages):
                if (res == 'cloud' and kind == 'cpu') or (scope == 'gateway' and res != t.gateway):
                    continue
                earliest = t.release_s + sum(durations[:s])
                latest = t.release_s + t.deadline_s - sum(durations[s + 1:])
                if kind == 'link':
                    latest -= sc.node(res).uplink_latency_s
                members.setdefault((kind, res), []).append((earliest, latest, amount, col))
    n = len(options)
    a_eq = coo_matrix((np.ones(n), ([r for r, _, _ in options], list(range(n)))),
                      shape=(len(task_ids), n)).tocsr()
    end = max(sc.tasks[i].release_s + sc.tasks[i].deadline_s for i in task_ids)
    starts = [k * sc.epoch_s for k in range(int(math.ceil(end / sc.epoch_s)) + 1)]
    ub_r, ub_c, ub_v, b_ub = [], [], [], []
    for (kind, res), items in members.items():
        spec = sc.node(res)
        rate = spec.mips if kind == 'cpu' else spec.uplink_bytes_s
        items.sort()
        ws = np.array([w[0] for w in items])
        own = {(round(w[0], 9), round(w[1], 9)) for w in items}
        grown = {(a, round(a + L, 9)) for a, _ in own for L in lengths}
        for a, b in sorted(set(_intervals(starts, lengths)) | own | grown):
            if b <= a:
                continue
            lo = np.searchsorted(ws, a - 1e-9)
            inside = []
            for s, e, amount, col in items[lo:]:
                if s >= b:
                    break
                if e <= b + 1e-9:
                    inside.append((amount, col))
            capacity = rate * (b - a)
            if not inside or sum(x for x, _ in inside) <= capacity:
                continue
            row = len(b_ub)
            ub_r.extend([row] * len(inside))
            ub_c.extend(col for _, col in inside)
            ub_v.extend(amount for amount, _ in inside)
            b_ub.append(capacity)
    a_ub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(len(b_ub), n)).tocsr() if b_ub else None
    res = linprog(np.array(cost), A_ub=a_ub, b_ub=np.array(b_ub) if b_ub else None, A_eq=a_eq,
                  b_eq=np.ones(len(task_ids)), bounds=(0, 1), method='highs',
                  options={'time_limit': time_limit_s})
    if res.status != 0:
        raise RuntimeError(f'LP not solved: {res.message}')
    for col, (_, i, o) in enumerate(options):
        x = res.x[col]
        if x <= 1e-12:
            continue
        t = sc.tasks[i]
        for c, p in sc.detection(t, o).items():
            det[c] += x * p
        total['fa'] += x * sc.false_alerts(t, o)
    total['objective'] += float(res.fun)
    total['options'] += n
    total['constraints'] += len(b_ub)
