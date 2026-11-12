# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Exercise the tuner's live-apply tiers on one example, headless.

    python scripts/live_check.py <example> [--solver <preset>] [--presets <file>]

Builds the example with the null viewer, steps a few frames, then applies through
``common.live`` (exactly what the tuner's Apply does):

1. ``sim.substeps`` one higher (the live timing path), then back;
2. every ``live`` solver setting with a nudged value, read back from the solver, then restored;
3. every ``model`` setting with its own value (exercises the array remap and the notification);
4. a solver swap with the current settings (``build_solver()`` or the plain one-liner).

Steps a few frames after each stage. Prints one ``LIVE`` line: ``LIVE OK <example> <preset>
tiers=...`` or ``LIVE FAIL <example> <preset> <stage>: <error>``. Exit code 1 on failure.
A timing, live or model change the example cannot absorb (the tuner then rebuilds) is not a
failure: it is listed as ``fallbacks=key:Error:...`` on the OK line. The swap stage must pass.
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import live, modes, presets


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _nudge(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.5 if value != 0.0 else 0.1
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("example")
    parser.add_argument("--solver")
    parser.add_argument("--presets")
    parser.add_argument("--frames", type=int, default=3, help="frames stepped after each stage")
    args = parser.parse_args()
    matches = glob.glob(str(ROOT / "shelf" / "*" / args.example / "example.py"))
    if not matches:
        sys.exit(f"no example named {args.example}")
    path = Path(matches[0])
    presets.select(args.solver, {}, args.presets)
    live.install_recorder()
    live.reset_record()
    outcome: list[str] = []

    def loop(example, _args):
        import warp as wp

        preset = example.preset
        name = preset.name
        settings = _flatten({k: v for k, v in preset.raw.items() if k in (*presets.SECTIONS, "entries")})
        tiers = {k: live.tier(example, k, preset) for k in settings}
        used: set[str] = set()

        def step(n):
            for _ in range(n):
                example.step()
            wp.synchronize_device()

        def reload(overrides):
            presets.select(name, overrides, args.presets)
            return presets.load(path)

        import newton

        def solvers():
            return {n: v for n, v in vars(example).items() if isinstance(v, newton.solvers.SolverBase)}

        # A failure in the timing, live or model stage is what the tuner survives by rebuilding, so it
        # is recorded as a fallback and the value restored; a failure in the swap stage (build_solver
        # or the plain swap) or anywhere else fails the check.
        fallbacks: list[str] = []

        def read_back(key):
            where, attr = live.LIVE_SOLVER[live._solver_kind(preset)][key[len("solver.") :]]
            if where == "solver":
                return getattr(example.solver, attr)
            if where == "coupling":
                return getattr(example.solver._coupling, attr)
            if where == "mjw_opt":
                return getattr(example.solver.mjw_model.opt, attr)
            return float(getattr(example.solver.mjw_model.opt, attr).numpy()[0])  # mjw_opt_fill

        def try_live(key, new, restore):
            """Apply ``new`` for ``key``, read it back, step, then restore; record a fallback on error."""
            try:
                live.apply(example, {key: new}, reload({key: new}))
                if key.startswith("solver."):
                    target = read_back(key)
                    if isinstance(new, float):
                        assert math.isclose(float(target), new, rel_tol=1e-6), f"read back {target}, set {new}"
                    else:
                        assert target == new, f"read back {target!r}, set {new!r}"
                elif key == "sim.substeps":
                    assert example.sim_substeps == new, "sim_substeps not applied"
                step(args.frames)
                used.add(tiers[key])
            except Exception as e:  # noqa: BLE001
                fallbacks.append(f"{key}:{type(e).__name__}:{str(e).splitlines()[0][:80]}")
            finally:
                try:
                    live.apply(example, {key: restore}, reload({}))
                except Exception:  # noqa: BLE001, S110 (a broken step; the swap stage rebuilds the solver anyway)
                    pass

        stage = "warmup"
        try:
            step(args.frames)
            if solvers() or hasattr(example, "build_solver"):  # multi-solver examples keep theirs in sub-objects
                stage = "timing"
                if tiers.get("sim.substeps") == "live":
                    base = settings["sim.substeps"]
                    try_live("sim.substeps", base + 1, base)
                stage = "live solver settings"
                for key, value in settings.items():
                    if tiers[key] == "live" and key.startswith("solver."):
                        try_live(key, _nudge(value), value)
                stage = "model settings"
                for key, value in settings.items():
                    if tiers[key] == "model":
                        try_live(key, value, value)
                stage = "swap"
                swap_keys = [k for k, t in tiers.items() if t == "swap"]
                before = solvers()
                graphs_before = live.graphs(example)
                if swap_keys:
                    live.apply(example, {swap_keys[0]: settings[swap_keys[0]]}, reload({}))
                else:
                    live._swap(example, reload({}))
                    live.recapture(example)
                after = solvers()
                same = [n for n, s in before.items() if after.get(n) is s]
                assert not same, f"not swapped: {', '.join(same)}"
                stale = [n for n, g in graphs_before.items() if getattr(example, n, None) is g]
                assert not stale, f"CUDA graph still runs the old solver: {', '.join(stale)}"
                used.add("swap")
                step(args.frames)
            rebuild_keys = sorted(k for k, t in tiers.items() if t == "rebuild")
            outcome.append(
                f"LIVE OK {args.example} {name} tiers={','.join(sorted(used)) or 'none'}"
                + (f" fallbacks={';'.join(fallbacks)}" if fallbacks else "")
                + (f" rebuild_only={','.join(rebuild_keys)}" if rebuild_keys else "")
            )
        except Exception as e:  # noqa: BLE001
            outcome.append(f"LIVE FAIL {args.example} {name} {stage}: {type(e).__name__}: {e}")

    modes.run_example(path, ["--viewer", "null", "--headless"], loop)
    print(outcome[0] if outcome else f"LIVE FAIL {args.example} ?: the example never reached newton.examples.run()")
    return 0 if outcome and outcome[0].startswith("LIVE OK") else 1


if __name__ == "__main__":
    sys.exit(main())
