"""Run the smokescreen scenario in iFogSim (``org.fogids.IdsStaged``).

The Java driver executes the staged tasks with iFogSim's processor-sharing CPUs and
FIFO uplinks; the policy decides in Python at every epoch with released tasks. The
policy sees the state of a shadow ``Engine`` driven by the same decisions (its own
prediction of the resources, as ``PSModel`` is for qubo-ps) and the alerts sampled
from the tasks iFogSim actually completed on time. The metrics use iFogSim's finish
times; the shadow's are returned too, to measure how far the scheduler's model is
from the reference simulator.
"""
import json
from pathlib import Path
import subprocess
import tempfile
import time

from . import bridge
from .smokescreen import Engine


def compile_staged():
    classpath = bridge.compile_driver()
    java = bridge._ifogsim()
    source = bridge.ROOT / 'java/org/fogids/IdsStaged.java'
    target = java.CLASSES / 'org/fogids/IdsStaged.class'
    if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
        subprocess.run(['javac', '-encoding', 'UTF-8', '-cp', classpath, '-d', str(java.CLASSES), str(source)],
                       check=True)
    return classpath


def _instance(sc):
    nodes = sc.topology.to_dict()['nodes']
    tasks = [{'id': t.id, 'gateway': t.gateway, 'release_s': t.release_s} for t in sc.tasks]
    return {'epoch_s': sc.epoch_s, 'horizon_s': sc.horizon_s, 'nodes': nodes, 'tasks': tasks}


def _assignment(task, stages):
    if not stages:
        return [task.id, 0.0, None, 0, 0.0]
    gw_mi = stages[0][2]
    if len(stages) == 1:
        return [task.id, gw_mi, task.gateway, 0, 0.0]
    links = [s for s in stages[1:-1] if s[0] == 'link']
    size = int(round(links[0][2])) if links else 0
    _, node, mi = stages[-1]
    return [task.id, gw_mi, node, size, mi]


def simulate_ifogsim(sc, policy, decision_time='none', drain_s=10.0, timeout_s=7200, log_path=None):
    """``decision_time``: 'none' (decisions take effect at the epoch), 'measured' (the
    policy's wall-clock decision time is charged to the simulated clock) or a number of
    seconds charged at every epoch."""
    if decision_time not in ('none', 'measured') and not isinstance(decision_time, (int, float)):
        raise ValueError("decision_time must be 'none', 'measured' or seconds")
    classpath = compile_staged()
    shadow = Engine(sc)
    shadow.emit_alerts = False  # alerts come from iFogSim's completions
    policy.reset(sc)
    tasks = {t.id: t for t in sc.tasks}
    decision_s = []
    end = None
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'instance.json'
        path.write_text(json.dumps(_instance(sc)), encoding='utf-8')
        log = open(log_path, 'w', encoding='utf-8') if log_path else subprocess.DEVNULL
        process = subprocess.Popen(
            ['java', '-Xmx2g', '-cp', classpath, 'org.fogids.IdsStaged', str(path), str(drain_s)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8', bufsize=1)
        try:
            started = time.perf_counter()
            for line in process.stdout:
                if time.perf_counter() - started > timeout_s:
                    raise bridge.SimulationError('Simulation exceeded its time limit')
                message = json.loads(line)
                if message['type'] == 'end':
                    end = message
                    break
                now = message['t']
                shadow._advance(now)
                for i, f in sorted(message['completed'], key=lambda c: c[1]):
                    a = shadow.sample_alert(tasks[int(i)], shadow.decisions.get(int(i)), f)
                    if a is not None:
                        shadow.alerts.append(a)
                batch = [tasks[i] for i in message['tasks']]
                # The policy plans with the delay it expects: the charged constant, or the
                # mean of its own decision times so far.
                if decision_time == 'measured':
                    shadow.decision_delay = sum(decision_s) / len(decision_s) if decision_s else 0.0
                elif decision_time != 'none':
                    shadow.decision_delay = float(decision_time)
                t0 = time.perf_counter()
                chosen = policy.decide(shadow, now, batch)
                elapsed = time.perf_counter() - t0
                decision_s.append(elapsed)
                delay = 0.0 if decision_time == 'none' else elapsed if decision_time == 'measured' else float(decision_time)
                # The shadow applies the decisions when iFogSim does: after the charged delay.
                executed = shadow.apply(now + delay, batch, chosen)
                reply = {'assignments': [_assignment(t, executed[t.id]) for t in batch], 'delay_s': delay}
                process.stdin.write(json.dumps(reply) + '\n')
                process.stdin.flush()
            process.stdin.close()
            code = process.wait(timeout=120)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            if log_path:
                log.close()
    if end is None or code != 0:
        raise bridge.SimulationError(f'IdsStaged exited with code {code}')
    shadow._advance(float('inf'))
    finish = {r['id']: r['finish_s'] for r in end['results'] if r['finish_s'] is not None}
    metrics = sc.evaluate(shadow.decisions, finish)
    errors = [abs(finish[i] - shadow.finish[i]) for i in finish if i in shadow.finish]
    metrics['shadow'] = sc.evaluate(shadow.decisions, shadow.finish)
    metrics['finish_error_mean_s'] = sum(errors) / len(errors) if errors else None
    metrics['finish_error_max_s'] = max(errors) if errors else None
    metrics['decision_mean_s'] = sum(decision_s) / len(decision_s) if decision_s else 0.0
    metrics['decision_p95_s'] = sorted(decision_s)[int(0.95 * (len(decision_s) - 1))] if decision_s else 0.0
    metrics['decision_max_s'] = max(decision_s) if decision_s else 0.0
    return metrics
