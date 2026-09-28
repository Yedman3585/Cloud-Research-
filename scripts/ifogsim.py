"""Compile and run iFogSim without an IDE or machine-wide installation."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/ifogsim"
CLASSES = ROOT / "build/ifogsim/classes"


def classpath():
    jars = sorted(p for p in (SOURCE / "jars").rglob("*.jar") if not p.name.endswith("-sources.jar"))
    return os.pathsep.join([str(CLASSES), *map(str, jars)])


def build():
    sources = sorted((SOURCE / "src").rglob("*.java"))
    if not sources:
        raise RuntimeError("iFogSim source missing. See README.md for installation.")
    if CLASSES.exists():
        shutil.rmtree(CLASSES)
    CLASSES.mkdir(parents=True, exist_ok=True)
    argfile = CLASSES.parent / "sources.txt"
    # Forward slashes: javac @argfiles treat backslashes inside quotes as escapes (Windows paths).
    argfile.write_text("\n".join('"' + p.as_posix() + '"' for p in sources) + "\n")
    # The upstream v2.0.0 GUI sources contain non-UTF-8 bytes.
    subprocess.run(["javac", "-encoding", "ISO-8859-1", "-cp", classpath(),
                    "-d", str(CLASSES), "@" + str(argfile)], cwd=SOURCE, check=True)
    print(f"Compiled {len(sources)} source files into {CLASSES}")


def run(class_name, timeout):
    if not (CLASSES / (class_name.replace(".", "/") + ".class")).is_file():
        raise RuntimeError("Example not compiled. Run the build command first.")
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    log = output / (class_name.rsplit(".", 1)[-1] + ".log")
    with log.open("w") as stream:
        result = subprocess.run(["java", "-Xmx2g", "-cp", classpath(), class_name],
                                cwd=SOURCE, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout)
    print(f"Exit code: {result.returncode}; log: {log}")
    print("\n".join(log.read_text(errors="replace").splitlines()[-25:]))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "run"])
    parser.add_argument("--class-name", default="org.fog.test.perfeval.VRGameFog")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    try:
        if args.command == "build":
            build()
            return 0
        return run(args.class_name, args.timeout)
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
