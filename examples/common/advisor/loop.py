# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""The Steel tuning loop: run an example's whole episode, read the facts, apply the fixes, repeat.

    python -m shelf trial EXAMPLE [--settings '{"sim": {...}, "solver": {...}}'] --out DIR
    python -m shelf autotune EXAMPLE [--start preset|defaults|FILE.json] [--max-trials 6] --out DIR

A trial writes a presets file holding one ``[steel]`` table with the trial's settings (``row_watermark`` added),
runs the example for its episode with the null viewer, records the episode facts plus the stability advice, and
runs the example's own ``shelf check`` under the same settings. ``autotune`` runs trials in fresh processes until
the advice has no fix left, then tries cheaper settings and keeps those that stay healthy.
"""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import tomlkit

from common import cli, modes, presets, thumbs

from .episode import EpisodeProbe, model_facts, start_state_facts, time_step
from .stability import cost_candidates, healthy, improved, no_worse, recommend_stability, summarize

DEFAULT_SECONDS, MAX_SECONDS = 10.0, 20.0


def run_trial(args, passthrough: list[str]) -> None:
    """One trial in this process: build the example under ``--settings`` and probe its episode."""
    path = cli.find(args.example)
    data = presets.read(path)
    settings = json.loads(args.settings) if args.settings else _preset_settings(data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = adapt_spellings(settings, steel_params())
    toml = (out / "presets.toml").resolve()
    toml.write_text(trial_toml(data, settings, row_watermark=True))
    extra = check_contract(data)
    presets.select("steel", {}, file=str(toml))
    os.environ["SHELF_PRESETS"] = str(toml)
    fps = float(settings.get("sim", {}).get("fps") or 60)
    frames = args.frames or episode_frames(fps, extra, args.seconds)
    loop = _Episode(args.example, settings, frames)
    t0 = time.perf_counter()
    error = None
    try:
        modes.run_example(path, ["--viewer", "null", "--quiet", *extra, *passthrough], loop)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            error = f"SystemExit: {exc.code}"[-600:]
    except BaseException as exc:  # noqa: BLE001 - a build failure is a trial result
        error = f"{type(exc).__name__}: {exc}"[-600:]
    if not loop.result:
        if loop.ran:
            error = error or "the episode wrote no result"
        else:
            error = error or "the example never called newton.examples.run()"
        facts = {
            "example": args.example,
            "settings": settings,
            "build_error": error,
            "solver_defaults": _safe(lambda: steel_params()["defaults"]) or {},
        }
        result = {
            "status": "error" if loop.ran else "build_error",
            "error": error,
            "settings": settings,
            "facts": facts,
            "advice": recommend_stability(facts),
            "summary": None,
        }
    else:
        result = loop.result
        if error:
            # The example raised after its episode (teardown, its own post-run test): the trial failed.
            result["status"] = "error"
            result["error"] = "; ".join(e for e in (result.get("error"), error) if e)
            if "facts" in result:
                result["facts"]["run_error"] = error
                result["advice"] = recommend_stability(result["facts"])
        if not args.no_check and result.get("status") == "ok":
            ok, note, elapsed = cli.run_check(args.example, path, "steel", args.timeout, str(toml))
            result["facts"]["check"] = {"ok": bool(ok), "note": note, "seconds": round(elapsed, 1)}
            result["advice"] = recommend_stability(result["facts"])
            result["summary"] = summarize(result["facts"]) | {"check_ok": bool(ok)}
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    result["newton_commit"] = _safe(cli.newton_commit)
    result["gpu"] = _safe(cli.gpu_name)
    result["attempt"] = getattr(args, "attempt", None)
    tmp = out / f"trial.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str))
    tmp.replace(out / "trial.json")  # atomic: a reader sees the whole result or none
    print(f"TRIAL {result.get('status')} {args.example} -> {out}/trial.json", flush=True)


def run_tune(args, passthrough: list[str]) -> None:
    """Stability trials until no fix is left, then cheaper candidates, each kept only if it stays healthy."""
    data = presets.read(cli.find(args.example))
    settings = start_settings(data, args.start)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trials: list[dict] = []
    results: list[dict] = []

    def trial(s: dict, phase: str) -> dict:
        k = len(trials)
        tdir = (out / f"trial{k}").resolve()
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "trial.json").unlink(missing_ok=True)  # a result left by an earlier run is never read as this one
        attempt = uuid.uuid4().hex
        cmd = [
            sys.executable,
            "-m",
            "shelf",
            "trial",
            args.example,
            "--settings",
            json.dumps(s),
            "--out",
            str(tdir),
            "--timeout",
            str(args.timeout),
            "--attempt",
            attempt,
        ]
        if args.seconds:
            cmd += ["--seconds", str(args.seconds)]
        if args.no_check:
            cmd += ["--no-check"]
        t = time.perf_counter()
        rc = _spawn(cmd + passthrough, tdir / "trial.log", args.timeout * 2)
        res = _load(tdir / "trial.json")
        problem = None
        if rc != 0:
            problem = f"trial process rc={rc}"
        elif res is None:
            problem = "the trial process wrote no trial.json"
        elif res.get("attempt") != attempt:
            problem = "trial.json belongs to another attempt"
        if problem:
            # A process that failed is a failed trial, whatever it wrote; its own result stays as evidence.
            res = {
                "status": "process_failed",
                "error": problem,
                "settings": s,
                "child_status": (res or {}).get("status"),
                "advice": {"actions": [], "flags": [["critical", problem]], "critical": [problem]},
                "summary": None,
            }
        res.setdefault("advice", {"actions": [], "flags": [["critical", res.get("error") or "no result"]]})
        rec = {
            "trial": k,
            "phase": phase,
            "settings": s,
            "status": res.get("status"),
            "rc": rc,
            "wall_s": round(time.perf_counter() - t, 1),
            "summary": res.get("summary"),
            "actions": [a["id"] for a in res["advice"].get("actions", [])],
            "flags": res["advice"].get("flags", []),
            "healthy": healthy(res),
        }
        trials.append(rec)
        results.append(res)
        print(
            f"trial {k} [{phase}]: {rec['status']} healthy={rec['healthy']} actions={rec['actions']} "
            f"summary={rec['summary']}",
            flush=True,
        )
        return res

    # Phase 1: apply the stability fixes until none is left; a fix with a metric that does not lower it is reverted.
    rejected: set[str] = set()
    res = trial(settings, "fix")
    while len(trials) < args.max_trials:
        acts = [a for a in res["advice"].get("actions", []) if a["id"] not in rejected]
        nxt = apply_actions(settings, acts)
        if not acts or nxt == settings:
            break
        r = trial(nxt, "fix")
        checked = [a for a in acts if a.get("metric")]
        if (
            checked
            and len(checked) == len(acts)
            and not healthy(r)
            and not any(improved(res.get("summary"), r.get("summary"), a["metric"]) for a in checked)
        ):
            rejected |= {a["id"] for a in checked}
            trials[-1]["reverted"] = [a["id"] for a in checked]
            continue  # stay on the settings before this fix
        settings, res = nxt, r
    for t, r in zip(trials, results, strict=True):
        t["healthy"] = t["healthy"] or healthy(r, rejected)
    first_healthy = next((t["trial"] for t in trials if t["healthy"]), None)
    # Phase 2: cheaper settings, each kept only if it stays healthy, no worse and faster.
    best, best_res = settings, res
    anchor = (res or {}).get("summary") or {}  # health bounds stay those of the first healthy settings
    tried: set[str] = set()
    while (
        args.cost_trials
        and best_res is not None
        and healthy(best_res, rejected)
        and len(trials) < args.max_trials + args.cost_trials
    ):
        cands = [
            c
            for c in cost_candidates(best_res["facts"], best)
            if json.dumps(c["settings"], sort_keys=True) not in tried
        ]
        if not cands:
            break
        cand = cands[0]
        tried.add(json.dumps(cand["settings"], sort_keys=True))
        s = apply_actions(best, [cand])
        r = trial(s, "cost:" + cand["id"])
        if healthy(r, rejected) and no_worse(anchor, r["summary"], best_res["summary"]):
            best, best_res = s, r
            trials[-1]["kept"] = True
    final = {
        "example": args.example,
        "start": args.start,
        "trials": trials,
        "final_settings": best,
        "final_trial": next((t["trial"] for t in reversed(trials) if t["settings"] == best), None),
        "trial_count": len(trials),
        "first_healthy_trial": first_healthy,
        "final_healthy": bool(best_res is not None and healthy(best_res, rejected)),
        "rejected_fixes": sorted(rejected),
    }
    (out / "tune.json").write_text(json.dumps(final, indent=1, default=str))
    print(
        f"TUNE {args.example}: {len(trials)} trials, first healthy at {first_healthy}, final {best} -> {out}/tune.json",
        flush=True,
    )


def run_advise(args) -> None:
    """Re-run the stability rules on a recorded trial.json."""
    d = json.loads(Path(args.path).read_text())
    facts = d.get("facts", d)
    advice = recommend_stability(facts)
    print(
        json.dumps(
            {"summary": summarize(facts), "actions": advice["actions"], "flags": advice["flags"]}, indent=1, default=str
        )
    )


def episode_frames(fps: float, check_args: list[str], seconds: float | None = None) -> int:
    """Frames to run: ``seconds``, else 10 s or the check's own ``--num-frames`` when longer (at most 20 s)."""
    if seconds is None:
        seconds = DEFAULT_SECONDS
        nf = _num_frames(check_args)
        if nf and nf / fps > DEFAULT_SECONDS:
            seconds = min(nf / fps, MAX_SECONDS)
    return int(round(seconds * fps))


def start_settings(data: dict, start: str) -> dict:
    """``preset``: the example's [steel] table; ``defaults``: the scene's frame rate and model settings, 4 substeps
    and every solver setting at Steel's default (the starting point a new scene gets); a path: a JSON settings dict."""
    if start == "preset":
        return _preset_settings(data)
    if start == "defaults":
        table = data.get("steel", {})
        sim = {k: v for k, v in table.get("sim", {}).items() if k == "fps"} | {"substeps": 4}
        out = {"sim": sim, "solver": {}}
        for section in ("model", "collision"):
            if table.get(section):
                out[section] = copy.deepcopy(table[section])
        return out
    return json.loads(Path(start).read_text())


def steel_params() -> dict:
    """SolverSteel's constructor parameters and the solve-mode / response spellings this Newton accepts."""
    import inspect
    import typing

    from newton.solvers import SolverSteel

    params = inspect.signature(SolverSteel.__init__).parameters
    choices = lambda name: list(typing.get_args(params[name].annotation)) if name in params else []  # noqa: E731
    plain = (bool, int, float, str)
    return {
        "names": sorted(p for p in params if p not in ("self", "model")),
        "defaults": {k: p.default for k, p in params.items() if isinstance(p.default, plain)},
        "solve_mode": choices("solve_mode"),
        "articulated_contact_response": choices("articulated_contact_response"),
    }


def adapt_spellings(settings: dict, params: dict) -> dict:
    """Write the rules' solve-mode names (serial, colored + propagation) in the spelling the installed Newton takes.

    Builds whose SolverSteel has no ``"serial"`` take ``"matrix_free"``, and ``"propagation_colored"`` for colored
    propagation."""
    out = copy.deepcopy(settings)
    solver = out.get("solver") or {}
    modes_ = params.get("solve_mode") or []
    if not modes_ or "serial" in modes_:
        return out
    mode = solver.get("solve_mode")
    if mode == "colored":
        solver["solve_mode"] = "matrix_free"
        solver["articulated_contact_response"] = "propagation_colored"
    elif mode == "serial":
        solver["solve_mode"] = "matrix_free"
    return out


def apply_actions(settings: dict, actions: list[dict]) -> dict:
    """Merge each action's ``settings`` ({"section.key": value}) into a copy of ``settings``."""
    out = copy.deepcopy(settings)
    for a in actions:
        for dotted, value in a.get("settings", {}).items():
            section, key = dotted.split(".", 1)
            if value is None:
                out.get(section, {}).pop(key, None)
            else:
                out.setdefault(section, {})[key] = value
    return out


def check_contract(data: dict) -> list[str]:
    """The Steel preset's check arguments without ``--set`` overrides: the scene and task arguments (world count,
    sizes, length) stay; settings overrides would make the check run other settings than the trial it judges."""
    return thumbs._split_check_args(cli.check_args(data, "steel"))[0]


def trial_toml(data: dict, settings: dict, row_watermark: bool) -> str:
    """A presets.toml with the example's metadata and one [steel] table holding ``settings``."""
    doc = tomlkit.document()
    doc["example"] = data.get("example", "example")
    doc["default"] = "steel"
    doc["check"] = {"args": check_contract(data)}
    doc["gallery"] = {"short": data.get("gallery", {}).get("short", "trial")}
    table = tomlkit.table()
    table["status"] = "works"
    for section in ("sim", "collision", "model"):
        if settings.get(section):
            table[section] = modes._inline(settings[section])
    table["solver"] = modes._inline(
        dict(settings.get("solver", {})) | ({"row_watermark": True} if row_watermark else {})
    )
    doc["steel"] = table
    return tomlkit.dumps(doc)


class _Episode:
    """Replacement for ``newton.examples.run``: start-state check, the episode under the probe, the advice."""

    def __init__(self, example_name: str, settings: dict, frames: int):
        self.name, self.settings, self.frames = example_name, settings, frames
        self.ran = False
        self.result: dict = {}

    def __call__(self, example, example_args) -> None:
        from newton.solvers import SolverSteel

        self.ran = True
        solver = getattr(example, "solver", None)
        if not isinstance(solver, SolverSteel):
            self.result = {
                "status": "error",
                "error": f"example.solver is {type(solver).__name__}, not SolverSteel",
                "settings": self.settings,
            }
            return
        model = example.model
        substeps = int(getattr(example, "sim_substeps", 0) or self.settings.get("sim", {}).get("substeps", 1))
        frame_dt = float(getattr(example, "frame_dt", 0.0) or 1.0 / float(self.settings.get("sim", {}).get("fps", 60)))
        substep_dt = float(getattr(example, "sim_dt", 0.0) or frame_dt / max(substeps, 1))
        pipe = getattr(example, "collision_pipeline", None)
        matcher = getattr(pipe, "contact_matcher", None) or getattr(pipe, "_contact_matcher", None)
        threshold = None
        if matcher is not None and getattr(matcher, "_pos_threshold_sq", None) is not None:
            threshold = float(matcher._pos_threshold_sq) ** 0.5
        start = start_state_facts(model, example.state_0)
        static = model_facts(model, solver)
        probe = EpisodeProbe(model, solver, substep_dt, substep_dt, threshold)
        probe.observe_initial(example.state_0)
        error = None
        f = -1
        t0 = time.perf_counter()
        try:
            for f in range(self.frames):
                ms = time_step(example.step)
                probe.update(example.state_0, getattr(example, "contacts", None), ms)
                h = probe.health
                if h.nonfinite_first is not None and h.frames > h.nonfinite_first[0] + 30:
                    break
        except Exception as exc:  # noqa: BLE001 - a step exception is a trial result
            error = f"{type(exc).__name__}: {exc}"[-600:]
        ep = probe.finish()
        facts = {
            "example": self.name,
            "settings": self.settings,
            "frames_planned": self.frames,
            "frames_done": f + 1,
            "solver_params": steel_params()["names"],
            "substeps": substeps,
            "frame_dt": frame_dt,
            "substep_dt": substep_dt,
            "start_state": start,
            "model": static,
            "episode": ep,
            "step_error": error,
            "probe_wall_s": round(time.perf_counter() - t0, 1),
            "solver_kwargs": _solver_kwargs(solver),
        }
        advice = recommend_stability(facts)
        self.result = {
            "status": "ok" if error is None else "step_error",
            "error": error,
            "settings": self.settings,
            "facts": facts,
            "advice": advice,
            "summary": summarize(facts),
        }


def _preset_settings(data: dict) -> dict:
    table = data.get("steel", {})
    return {k: copy.deepcopy(table[k]) for k in ("sim", "collision", "solver", "model") if k in table}


def _solver_kwargs(solver) -> dict:
    """The solver options the rules compare against (public attribute, then the private one)."""
    keys = (
        "solve_mode",
        "iterations",
        "baumgarte",
        "articulated_contact_response",
        "friction_anchor_beta",
        "contact_warmstart",
        "contact_regularization",
        "velocity_iterations",
        "enable_joint_limits",
        "enable_sleeping",
        "enable_restitution",
        "nesterov_momentum",
        "contact_torsion_device",
        "propagation_color_order",
        "articulation_constraint_max",
        "free_body_constraint_max",
        "propagation_constraint_max",
        "relaxation",
    )
    out = {}
    for k in keys:
        for name in (k, "_" + k):
            if hasattr(solver, name):
                v = getattr(solver, name)
                if isinstance(v, (bool, int, float, str, type(None))):
                    out[k] = v
                    break
    return out


def _num_frames(args: list[str]) -> int | None:
    for i, a in enumerate(args):
        if a == "--num-frames" and i + 1 < len(args):
            return int(args[i + 1])
        if a.startswith("--num-frames="):
            return int(a.split("=", 1)[1])
    return None


def _spawn(cmd: list[str], log: Path, timeout: int) -> int | str:
    """Run one trial in its own process group; on return every process of that group (the check too) has ended."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=cli.ROOT, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc: int | str = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = "timeout"
        _end_group(proc)
        return rc


def _end_group(proc: subprocess.Popen, grace: float = 10.0) -> None:
    """SIGTERM, then SIGKILL, the process group ``proc`` leads, and wait until none of it is left."""
    for sig, wait_s in ((signal.SIGTERM, grace), (signal.SIGKILL, 60.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            proc.poll()  # reap the leader so the group can empty
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
    proc.wait()


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - provenance only
        return None
