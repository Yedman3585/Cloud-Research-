"""Scheduler-side replica of the simulator's resource model.

The online simulator (iFogSim with CloudSim time sharing) serves the tasks on a
node by *egalitarian processor sharing*: n running tasks each progress at MIPS / n,
and a task that arrives later slows down every task already running. Uplinks are
FIFO: a transfer occupies the link for bytes / bandwidth and then adds the
propagation latency. ``QueueModel`` in ``policies.py`` treats the CPU as a FIFO
backlog, which misses the slow-down; under load its latency predictions are
uncorrelated or inversely correlated with the simulator (``docs/ids-place.md`` §9).

``PSModel`` keeps, for every node, the tasks it has committed there and evolves them
with the same processor-sharing rule, so it can predict

* the finish time of a candidate placement, and
* how much that placement delays the tasks already committed to the node
  (``estimate`` returns their new finish times).

It is built only from the policy's own decisions and the snapshot (completions,
resident models); it cannot know tasks of future epochs.

Processor sharing has a closed form that makes it a quadratic objective: for tasks
that start together with works a_i on one node, the sum of their finish times is
``sum_i a_i + 2 * sum_{i<j} min(a_i, a_j)`` (in units of work / MIPS). With binary
placement variables y_i this is ``sum_i a_i y_i + 2 sum_{i<j} min(a_i, a_j) y_i y_j``,
an exact QUBO term (``ps_sum_of_finish_times`` and its test).
"""


def ps_finish_times(mips, start, active, arrivals):
    """Processor-sharing finish times.

    ``active``: {key: remaining MI} running at time ``start``; ``arrivals``: list of
    (arrival_time, key, work_mi) with arrival_time >= start. Returns {key: finish}.
    """
    active = dict(active)
    pending = sorted(arrivals, key=lambda a: (a[0], str(a[1])))
    finish, t, k = {}, start, 0
    while active or k < len(pending):
        if not active:
            t = max(t, pending[k][0])
        while k < len(pending) and pending[k][0] <= t + 1e-12:
            _, key, work = pending[k]
            active[key] = work
            k += 1
        if not active:
            continue
        rate = mips / len(active)
        smallest = min(active.values())
        t_done = t + smallest / rate
        t_next = pending[k][0] if k < len(pending) else float('inf')
        t_event = min(t_done, t_next)
        served = (t_event - t) * rate
        for key in list(active):
            active[key] -= served
            if active[key] <= 1e-9 * max(1.0, mips):
                finish[key] = t_event
                del active[key]
        t = t_event
    return finish


def ps_sum_of_finish_times(works):
    """Sum of processor-sharing finish times (in work units) of tasks starting together:
    sum_i a_i + 2 sum_{i<j} min(a_i, a_j)."""
    a = sorted(works)
    return sum(a) + 2 * sum(a[i] * (len(a) - 1 - i) for i in range(len(a)))


class _Node:
    """Committed tasks on one node: running ones (remaining MI at ``t``) and ones still
    in transit (arrival time, MI)."""

    def __init__(self, mips):
        self.mips = mips
        self.t = 0.0
        self.active = {}
        self.pending = []  # (arrival, task_id, work)

    def advance(self, now):
        """Evolve the state to time ``now`` under processor sharing."""
        if now <= self.t:
            return
        arrivals = [a for a in self.pending if a[0] <= now]
        later = [a for a in self.pending if a[0] > now]
        active = dict(self.active)
        t = self.t
        pending = sorted(arrivals)
        k = 0
        while t < now - 1e-12:
            while k < len(pending) and pending[k][0] <= t + 1e-12:
                active[pending[k][1]] = pending[k][2]
                k += 1
            t_next = pending[k][0] if k < len(pending) else now
            if not active:
                t = min(t_next, now)
                continue
            rate = self.mips / len(active)
            t_done = t + min(active.values()) / rate
            t_event = min(t_done, t_next, now)
            served = (t_event - t) * rate
            for key in list(active):
                active[key] -= served
                if active[key] <= 1e-9 * max(1.0, self.mips):
                    del active[key]
            t = t_event
        for _, key, work in pending[k:]:
            active[key] = work
        self.active, self.pending, self.t = active, later, now

    def finishes(self, extra=None):
        """Predicted finish times of every committed task (and ``extra`` = (arrival,
        key, work) if given)."""
        arrivals = list(self.pending) + ([extra] if extra else [])
        return ps_finish_times(self.mips, self.t, self.active, arrivals)

    def remove(self, task_id):
        self.active.pop(task_id, None)
        self.pending = [a for a in self.pending if a[1] != task_id]


class PSModel:
    """FIFO uplinks and processor-sharing CPUs, mirroring the simulator."""

    def __init__(self, instance):
        self.p = instance
        self.link_free = {n.name: 0.0 for n in instance.nodes}
        self.nodes = {n.name: _Node(n.mips) for n in instance.nodes}
        self.resident = {n.name: list(n.resident) for n in instance.nodes}  # LRU order
        self.placed = {}  # task id -> node

    def sync(self, snapshot):
        """Drop reported completions and take the simulator's resident models."""
        for task_id, _ in snapshot.completed:
            node = self.placed.pop(task_id, None)
            if node is not None:
                self.nodes[node].remove(task_id)
        for name, state in snapshot.nodes.items():
            self.resident[name] = list(state['resident'])
        for state in self.nodes.values():
            state.advance(snapshot.t)

    def _transfer(self, task, node, now):
        t, done = now, []
        for link in self.p.path_links(task.gateway, node):
            spec = self.p.node(link)
            start = max(t, self.link_free[link])
            end = start + task.input_bytes / spec.uplink_bytes_s
            done.append((link, end))
            t = end + spec.uplink_latency_s
        return t, done

    def work(self, task, node, model):
        m, mips = self.p.model(model), self.p.node(node).mips
        return m.work_mi(task.n_flows) + (0.0 if model in self.resident[node] else m.load_s * mips)

    def estimate(self, task, node, model, now):
        """Return (finish, work, link completions, finish times of the node's other tasks
        with this task added)."""
        arrival, done = self._transfer(task, node, now)
        work = self.work(task, node, model)
        state = self.nodes[node]
        state.advance(now)
        finishes = state.finishes((arrival, task.id, work))
        own = finishes.pop(task.id)
        return own, work, done, finishes

    def finishes(self, node, now):
        """Current predicted finish times of the tasks committed to ``node``."""
        state = self.nodes[node]
        state.advance(now)
        return state.finishes()

    def commit(self, task, node, model, now):
        own, work, done, _ = self.estimate(task, node, model, now)
        arrival, _ = self._transfer(task, node, now)
        for link, end in done:
            self.link_free[link] = end
        self.nodes[node].pending.append((arrival, task.id, work))
        self.placed[task.id] = node
        self._touch(node, model)
        return own

    def _touch(self, node, model):
        """Least-recently-used residency within node memory, as in the simulator."""
        loaded = self.resident[node]
        if model in loaded:
            loaded.remove(model)
        loaded.append(model)
        memory = self.p.node(node).memory_mb
        while sum(self.p.model(m).memory_mb for m in loaded) > memory and len(loaded) > 1:
            loaded.pop(0)
