"""Epoch-based co-simulation: iFogSim executes, a Python policy decides.

The Java driver (``org.fogids.IdsOnline``) runs as a child process. At every epoch
in which tasks were released it sends a state snapshot as one JSON line on its
stdout and blocks until the policy's assignments arrive on its stdin, so
simulated time stands still while Python decides. Decision time is charged to
the simulation clock according to ``decision_time``:

* ``'none'``      assignments take effect immediately (placement-only effect);
* ``'measured'``  the policy's measured wall-clock time is charged;
* a number        a fixed delay in seconds (deterministic experiments).
"""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
JAVA_SOURCES = ['IdsBatch.java', 'IdsOnline.java']


@dataclass
class Snapshot:
    t: float
    epoch: int
    tasks: list  # Task objects released since the previous snapshot
    completed: list  # [(task_id, finish_s)] since the previous snapshot
    nodes: dict  # name -> {"outstanding": int, "resident": [model, ...]}


class SimulationError(RuntimeError):
    pass


def _ifogsim():
    spec = importlib.util.spec_from_file_location('ifogsim_build', ROOT / 'scripts/ifogsim.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_driver():
    """Compile the Java drivers when missing or older than their sources."""
    java = _ifogsim()
    if not (java.CLASSES / 'org/fog/entities/FogDevice.class').exists():
        java.build()
    sources = [ROOT / 'java/org/fogids' / s for s in JAVA_SOURCES]
    target = java.CLASSES / 'org/fogids/IdsOnline.class'
    if not target.exists() or max(s.stat().st_mtime for s in sources) > target.stat().st_mtime:
        subprocess.run(['javac', '-encoding', 'UTF-8', '-cp', java.classpath(), '-d', str(java.CLASSES),
                        *map(str, sources)], check=True)
    return java.classpath()


def simulate(instance, policy, decision_time='none', drain_s=30.0, log_path=None, timeout_s=600):
    """Run one instance under one policy; returns the driver's end message plus timings."""
    if decision_time not in ('none', 'measured') and not isinstance(decision_time, (int, float)):
        raise ValueError("decision_time must be 'none', 'measured' or a number of seconds")
    classpath = compile_driver()
    policy.reset(instance)
    decision_s = []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'instance.json'
        instance.save(path)
        log = open(log_path, 'w', encoding='utf-8') if log_path else subprocess.DEVNULL
        process = None
        try:
            process = subprocess.Popen(
                ['java', '-Xmx2g', '-cp', classpath, 'org.fogids.IdsOnline', str(path), str(drain_s)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, encoding='utf-8', bufsize=1)
            started = time.perf_counter()
            end = None
            for line in process.stdout:
                if time.perf_counter() - started > timeout_s:
                    process.kill()
                    raise SimulationError('Simulation exceeded its time limit')
                message = json.loads(line)
                if message['type'] == 'end':
                    end = message
                    break
                snapshot = Snapshot(
                    t=message['t'], epoch=message['epoch'],
                    tasks=[instance.tasks[i] for i in message['tasks']],
                    completed=[(int(i), f) for i, f in message['completed']], nodes=message['nodes'])
                t0 = time.perf_counter()
                assignments = policy.decide(snapshot)
                elapsed = time.perf_counter() - t0
                decision_s.append(elapsed)
                delay = 0.0 if decision_time == 'none' else elapsed if decision_time == 'measured' else float(decision_time)
                try:
                    process.stdin.write(json.dumps({'assignments': [list(a) for a in assignments],
                                                    'delay_s': delay}) + '\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    break  # the driver rejected the reply and exited; reported below
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            code = process.wait(timeout=60)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
            if log_path:
                log.close()
    if end is None or code != 0:
        detail = Path(log_path).read_text(encoding='utf-8')[-2000:] if log_path else 'set log_path for details'
        raise SimulationError(f'Simulator exited with code {code} before finishing:\n{detail}')
    end['decision_s'] = decision_s
    end['policy'] = policy.name
    end['decision_time'] = decision_time
    return end
