# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Run scripts/live_check.py for every preset marked ``works`` of the given (or all) examples.

    python scripts/live_sweep.py [--out results.txt] [--timeout 600] [example ...]

Prints one ``LIVE ...`` line per (example, preset) and a summary; the lines also go to ``--out``.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("examples", nargs="*")
    parser.add_argument("--out", default="live_sweep.txt")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    jobs: list[tuple[str, str]] = []
    for f in sorted(glob.glob(str(ROOT / "shelf" / "*" / "*" / "presets.toml"))):
        name = Path(f).parent.name
        if args.examples and name not in args.examples:
            continue
        with open(f, "rb") as fh:
            data = tomllib.load(fh)
        if data.get("default") == "none":
            continue
        for preset, table in data.items():
            if isinstance(table, dict) and table.get("status") == "works":
                jobs.append((name, preset))
    ok = fail = 0
    with open(args.out, "w") as out:
        for name, preset in jobs:
            cmd = [PYTHON, str(ROOT / "scripts" / "live_check.py"), name, "--solver", preset]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, cwd=ROOT, check=False)
                lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("LIVE ")]
                line = (
                    lines[-1]
                    if lines
                    else f"LIVE FAIL {name} {preset} crashed: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'no output'}"
                )
            except subprocess.TimeoutExpired:
                line = f"LIVE FAIL {name} {preset} timeout after {args.timeout}s"
            ok += line.startswith("LIVE OK")
            fail += not line.startswith("LIVE OK")
            print(line, flush=True)
            out.write(line + "\n")
            out.flush()
        summary = f"SUMMARY {ok} ok, {fail} failed, {len(jobs)} runs"
        print(summary)
        out.write(summary + "\n")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
