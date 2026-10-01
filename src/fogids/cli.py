"""Thin CLI for launching the Java simulator from Python."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check local runtime and simulator checkout")
    sub.add_parser("build", help="Compile the pinned iFogSim checkout")
    run = sub.add_parser("run", help="Run an upstream example; not a custom IDS simulation")
    run.add_argument("--class-name", default="org.fog.test.perfeval.VRGameFog")
    run.add_argument("--timeout", type=int, default=120)
    demo = sub.add_parser("ids-demo", help="Run a tiny QUBO placement experiment in iFogSim")
    demo.add_argument("--config", type=Path, default=ROOT / "configs/ids-small.json")
    demo.add_argument("--output", type=Path, default=ROOT / "artifacts/ids-small")
    gen = sub.add_parser("generate", help="Generate an IDS-PLACE instance from a generator config")
    gen.add_argument("--config", type=Path, default=ROOT / "configs/ids-place-small.json")
    gen.add_argument("--output", type=Path, default=None)
    gen.add_argument("--seed", type=int, default=None, help="Override the config seed")
    desc = sub.add_parser("describe", help="Print load and size statistics of an instance file")
    desc.add_argument("instance", type=Path)
    sim = sub.add_parser("simulate", help="Run an IDS-PLACE instance online in iFogSim under policies")
    source = sim.add_mutually_exclusive_group()
    source.add_argument("--instance", type=Path, help="Instance JSON file")
    source.add_argument("--config", type=Path, default=ROOT / "configs/ids-place-small.json",
                        help="Generator config (used when --instance is not given)")
    sim.add_argument("--seed", type=int, default=None)
    sim.add_argument("--policy", action="append", default=None,
                     help="greedy-finish or <edge|fog|cloud>-<model>; repeatable (default: four baselines)")
    sim.add_argument("--decision-time", default="none", help="none, measured, or a fixed delay in seconds")
    sim.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "simulate":
        from .simulate import run_cli
        return run_cli(args, ROOT)
    if args.command == "generate":
        from .generator import generate, load_config
        config = load_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        instance = generate(config)
        output = args.output or ROOT / "artifacts/instances" / f"{config['name']}-s{config['seed']}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        instance.save(output)
        print(json.dumps(instance.summary(), indent=2))
        print(f"Instance: {output}")
        return 0
    if args.command == "describe":
        from .problem import Instance
        print(json.dumps(Instance.load(args.instance).summary(), indent=2))
        return 0
    if args.command == "ids-demo":
        from .experiment import run as run_experiment
        run_experiment(args.config, args.output)
        return 0
    if args.command == "doctor":
        checks = {
            "java": shutil.which("java"),
            "javac": shutil.which("javac"),
            "ifogsim_source": (ROOT / "vendor/ifogsim/src").is_dir(),
            "compiled": (ROOT / "build/ifogsim/classes/org/fog/entities/FogDevice.class").is_file(),
        }
        print(json.dumps(checks, indent=2))
        return 0 if all(checks.values()) else 1
    script = ROOT / "scripts/ifogsim.py"
    import sys
    command = [sys.executable, str(script), args.command]
    if args.command == "run":
        command += ["--class-name", args.class_name, "--timeout", str(args.timeout)]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
