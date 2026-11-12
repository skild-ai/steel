# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Previews for the example browser: one frame of each example, rendered headless and cached per look.

``python -m shelf thumbs`` renders them, every example or the ones named (``--missing``: only those
not cached yet). ``python -m shelf play`` with no example opens the browser and renders the missing
ones in the background, so the first visit shows cards filling in. Each example renders in its own
process, headless, with the default preset and the look given, at the frame its presets.toml
``[gallery] frame`` names (default :data:`presets.DEFAULT_THUMB_FRAME`), and the frame is saved as a
:data:`SIZE` PNG under ``$XDG_CACHE_HOME/steel-shelf/thumbs/<look>/<example>.png``. A render that
fails leaves ``<example>.error`` with the reason, so the browser can say so and nothing retries it
until ``--force``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import presets

ROOT = Path(__file__).resolve().parents[1]
SIZE = (640, 360)  # the cached preview, 16:9 like the viewer's window
DEFAULT_JOBS = 2  # examples rendering at once; each is a process with its own CUDA context
DEFAULT_TIMEOUT = 240  # seconds per example
# The whole shelf on an RTX 3090 with DEFAULT_JOBS, what the first visit says: measured 12.7 min for 129 examples on
# 2026-11-04 before the slowest previews were trimmed to earlier frames (about 11 min after).
FIRST_BUILD_MINUTES = 11
PROGRESS = "progress.json"
BUILDER_PID = "builder.pid"


def cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "steel-shelf" / "thumbs"


def folder(look: str) -> Path:
    return cache_root() / look


def png(look: str, example: str) -> Path:
    return folder(look) / f"{example}.png"


def error_file(look: str, example: str) -> Path:
    return folder(look) / f"{example}.error"


# --- one example, in this process -------------------------------------------------------------------


class Thumb:
    """Replacement for ``newton.examples.run`` that steps to ``frame``, grabs the frame and stops."""

    def __init__(self, out: Path, frame: int):
        self.out = out
        self.frame = frame
        self.ran = False  # stays False when the example never calls newton.examples.run()

    def __call__(self, example, args) -> None:
        from PIL import Image

        self.ran = True
        viewer = example.viewer
        if hasattr(viewer, "hide_loading_splash"):
            viewer.hide_loading_splash()
        for _ in range(max(1, self.frame)):  # as Newton's loop: a step, then the render that shows it
            example.step()
            example.render()
        image = viewer.get_frame().numpy()  # height x width x 3
        self.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out.with_suffix(".part")
        Image.fromarray(image).resize(SIZE, Image.LANCZOS).save(tmp, format="PNG")
        tmp.replace(self.out)
        viewer.close()


def render(path: Path, out: Path, frame: int, look: str | None) -> None:
    """Run the example at ``path`` headless with its default preset and save frame ``frame`` to ``out``.

    The example's ``[check]`` arguments are passed, as ``shelf check`` does, so the scene is the one
    the presets were checked with.
    """
    from common import cli, modes

    data = presets.read(path)
    extra, overrides = _split_check_args(cli.check_args(data, data.get("default", "")))
    presets.select(None, overrides)  # the default preset, with the overrides `shelf check` would apply
    loop = Thumb(out, frame)
    modes.run_example(path, ["--viewer", "gl", "--headless", "--quiet", *extra], loop=loop, look=look)
    if not loop.ran:
        raise RuntimeError("the example runs its own loop instead of newton.examples.run(), so it has no preview")


def _split_check_args(args: list[str]) -> tuple[list[str], dict]:
    """The example's own arguments and the ``--set section.key=value`` overrides among them, which
    ``shelf run`` parses itself (``--num-frames`` stays: the loop above is ours, so it is harmless)."""
    rest, overrides = [], {}
    it = iter(args)
    for a in it:
        if a == "--viewer":  # the preview renders with the GL viewer whatever `check` uses
            next(it, None)
        elif a.startswith("--viewer="):
            pass
        elif a == "--set":
            key, value = presets.parse_override(next(it, ""))
            overrides[key] = value
        elif a.startswith("--set="):
            key, value = presets.parse_override(a[len("--set="):])
            overrides[key] = value
        else:
            rest.append(a)
    return rest, overrides


# --- the build, one process per example --------------------------------------------------------------


def frame_of(path: Path) -> int:
    """The frame the example's preview shows: ``[gallery] frame``, else the default."""
    try:
        entry = presets.gallery(path)
    except presets.PresetError:
        entry = None
    return entry["frame"] if entry else presets.DEFAULT_THUMB_FRAME


def missing(targets: dict[str, Path], look: str) -> dict[str, Path]:
    """The examples of ``targets`` (name -> example.py) with neither a preview nor a recorded failure."""
    return {n: p for n, p in targets.items() if not png(look, n).exists() and not error_file(look, n).exists()}


def _cuda_devices() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
        return max(1, sum(1 for ln in out.splitlines() if ln.startswith("GPU ")))
    except (OSError, subprocess.SubprocessError):
        return 1


def _write_progress(look: str, state: dict[str, Any]) -> None:
    state["updated"] = time.time()
    path = folder(look) / PROGRESS
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def build(
    targets: dict[str, Path],
    look: str,
    jobs: int = DEFAULT_JOBS,
    timeout: int = DEFAULT_TIMEOUT,
    force: bool = False,
    quiet: bool = False,
) -> int:
    """Render previews for ``targets`` (name -> example.py), ``jobs`` at a time; returns how many failed.

    Progress goes to ``progress.json`` in the look's cache folder for the browser to show. On a box
    with several GPUs the jobs are spread over them.
    """
    out_dir = folder(look)
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for name in targets:
            png(look, name).unlink(missing_ok=True)
            error_file(look, name).unlink(missing_ok=True)
    todo = list(targets.items())
    state: dict[str, Any] = {
        "look": look, "total": len(todo), "done": 0, "failed": 0, "started": time.time(),
        "running": [], "seconds": {}, "errors": {}, "finished": False,
    }  # fmt: skip
    _write_progress(look, state)
    (out_dir / BUILDER_PID).write_text(str(os.getpid()))
    gpus = _cuda_devices()
    running: dict[str, tuple[subprocess.Popen, float, int]] = {}  # name -> (process, start, gpu)
    used_gpus: list[int] = []
    failures = 0
    try:
        while todo or running:
            while todo and len(running) < max(1, jobs):
                name, path = todo.pop(0)
                gpu = next((g for g in range(gpus) if used_gpus.count(g) < max(1, -(-jobs // gpus))), 0)
                env = dict(os.environ)
                if gpus > 1:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                cmd = [
                    sys.executable, "-m", "shelf", "_thumb", name,
                    "--out", str(png(look, name)), "--frame", str(frame_of(path)), "--look", look,
                ]  # fmt: skip
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                running[name] = (proc, time.perf_counter(), gpu)
                used_gpus.append(gpu)
                state["running"] = list(running)
                _write_progress(look, state)
            time.sleep(0.25)
            for name in list(running):
                proc, start, gpu = running[name]
                elapsed = time.perf_counter() - start
                if proc.poll() is None and elapsed < timeout:
                    continue
                if proc.poll() is None:
                    proc.kill()
                    output = f"timed out after {timeout} s"
                    code = -1
                else:
                    output, code = (proc.stdout.read() if proc.stdout else ""), proc.returncode
                del running[name]
                used_gpus.remove(gpu)
                state["done"] += 1
                state["seconds"][name] = round(elapsed, 1)
                if code != 0 or not png(look, name).exists():
                    failures += 1
                    state["failed"] += 1
                    lines = [ln for ln in output.strip().splitlines() if ln.strip() and "load on device" not in ln]
                    reason = (lines[-1] if lines else f"exit code {code}")[:300]
                    if code == 0:  # the example ended the process itself before the frame was reached
                        reason = f"the example ended on its own before frame {frame_of(targets[name])}: {reason}"
                    error_file(look, name).write_text(reason + "\n")
                    state["errors"][name] = reason
                    if not quiet:
                        print(f"{name:45s} FAILED {elapsed:6.1f} s  {reason}", flush=True)
                elif not quiet:
                    print(f"{name:45s} ok     {elapsed:6.1f} s", flush=True)
                state["running"] = list(running)
                _write_progress(look, state)
    finally:
        state["finished"] = True
        state["running"] = []
        _write_progress(look, state)
        (out_dir / BUILDER_PID).unlink(missing_ok=True)
    return failures


# --- the browser's view of the build ----------------------------------------------------------------


def builder_alive(look: str) -> bool:
    pid_file = folder(look) / BUILDER_PID
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def status(look: str) -> dict[str, Any] | None:
    """The last progress written by a build of ``look``, with ``alive`` for whether it still runs."""
    path = folder(look) / PROGRESS
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    state["alive"] = builder_alive(look) and not state.get("finished", False)
    done = state.get("done", 0)
    seconds = sum(state.get("seconds", {}).values())
    left = state.get("total", 0) - done
    rate = seconds / done if done else 0.0
    state["eta_s"] = (left * rate / max(1, DEFAULT_JOBS)) if done else None
    return state


def start_background(look: str) -> tuple[int, int]:
    """Render the previews ``look`` lacks in a detached process, unless one is already at it.

    Returns ``(missing, total)``: how many previews are being rendered (0 when nothing is missing or a
    builder already runs) and how many examples the shelf has.
    """
    from common import cli

    targets = cli.examples()
    todo = missing(targets, look)
    if not todo or builder_alive(look):
        return 0, len(targets)
    folder(look).mkdir(parents=True, exist_ok=True)
    log = (folder(look) / "build.log").open("a")
    subprocess.Popen(
        [sys.executable, "-m", "shelf", "thumbs", "--missing", "--look", look],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )  # fmt: skip
    return len(todo), len(targets)
