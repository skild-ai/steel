# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""The render audit's scripts (scripts/render_audit): what may never be graded ok."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[2]
SCRIPTS = EXAMPLES / "scripts" / "render_audit"


def _module(name):
    spec = importlib.util.spec_from_file_location(f"render_audit_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


collect = _module("collect")
needs_ffmpeg = pytest.mark.skipif(shutil.which(collect.FFMPEG) is None, reason="ffmpeg not found")


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def numpy(self):
        return self.values


def test_invalid_contact_geometry_is_counted_and_unstable():
    pytest.importorskip("pyglet")
    render_scene = _module("render_scene")
    metrics = render_scene.Metrics.__new__(render_scene.Metrics)
    metrics.contacts = type("Contacts", (), {})()
    for name, value in {
        "rigid_contact_count": [1],
        "rigid_contact_shape0": [0],
        "rigid_contact_shape1": [1],
        "rigid_contact_point0": [[np.nan, 0.0, 0.0]],
        "rigid_contact_point1": [[0.0, 0.0, 0.0]],
        "rigid_contact_normal": [[0.0, 0.0, 1.0]],
        "rigid_contact_margin0": [0.0],
        "rigid_contact_margin1": [0.0],
    }.items():
        setattr(metrics.contacts, name, _Array(value))
    metrics.nb, metrics.fps, metrics.shape_body = 1, 60.0, np.array([0, -1])
    metrics.pen_series, metrics.contact_series = [], []
    metrics.invalid_contact_frames, metrics.invalid_contacts, metrics.invalid_contact_first_frame = 0, 0, None
    state = type("State", (), {"body_q": _Array([[0, 0, 0, 0, 0, 0, 1.0]])})()
    metrics._contacts(0, state)
    assert metrics.invalid_contact_frames == 1 and metrics.invalid_contacts == 1
    res = {"loop_ran": True, "fps": 60.0, "metrics": {"invalid_contact_frames": 1, "invalid_contacts": 1}}
    grade, reasons = collect.grade({}, res, {"result": "works"})
    assert grade == "unstable" and "non-finite geometry" in reasons[0]


def test_an_inherited_override_never_reaches_the_audited_preset(monkeypatch):
    pytest.importorskip("pyglet")
    monkeypatch.chdir(EXAMPLES)
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from common import cli, presets

    render_scene = _module("render_scene")
    monkeypatch.setenv("SHELF_SET", json.dumps({"solver.iterations": 123}))
    monkeypatch.setattr(presets, "_selected", None)
    monkeypatch.setattr(presets, "_overrides", {})
    monkeypatch.setattr(presets, "_file", None)
    cleared = render_scene.select_committed("steel")
    path = cli.find("basic_pendulum")
    assert "SHELF_SET" in cleared and "SHELF_SET" not in os.environ
    assert presets.load(path).solver.need("iterations") == presets.read(path)["steel"]["solver"]["iterations"]


def _run_dir(tmp_path, *, attempt="a1", metrics_attempt="a1", render_rc=0, check_rc=0, clip="valid"):
    run = tmp_path / "run"
    out = run / "out" / "s"
    out.mkdir(parents=True)
    entry = {"scene": "s", "title": "s", "short": "", "path": "shelf/x/s", "category": "x", "preset": "steel",
             "kind": "steel", "status": "works", "checked": "", "settings": {}, "check_args": [],
             "sim_seconds": 1.0, "speed": 1.0}  # fmt: skip
    (run / "plan.json").write_text(json.dumps({"commit": "abc", "newton_pin": "n", "render": [entry],
                                               "skipped": [], "excluded": []}))  # fmt: skip
    (out / "attempt").write_text(attempt + "\n")
    metrics = {"loop_ran": True, "attempt": metrics_attempt, "fps": 60.0, "error": None, "metrics": {}}
    (out / "metrics.json").write_text(json.dumps(metrics))
    (out / "render.log").write_text(f"RENDERED s\nRENDER_RC {render_rc}\n")
    (out / "check.log").write_text(f"s steel works 1.0 s\nCHECK_RC {check_rc}\n")
    if clip == "valid":
        subprocess.run([collect.FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=0.5",
                        "-pix_fmt", "yuv420p", str(out / "clip_raw.mp4")], check=True)  # fmt: skip
    elif clip == "corrupt":
        (out / "clip_raw.mp4").write_bytes(b"\0" * 64)
    return run


def _collect(run):
    subprocess.run([sys.executable, str(SCRIPTS / "collect.py"), str(run)], check=True, capture_output=True)
    results = json.loads((run / "results.json").read_text())
    return results, results["scenes"][0]


@needs_ffmpeg
def test_a_current_successful_render_is_ok(tmp_path):
    results, scene = _collect(_run_dir(tmp_path))
    assert scene["grade"] == "ok" and results["rendered"] == 1 and scene["files"]["video"] == "media/s.mp4"


@needs_ffmpeg
@pytest.mark.parametrize(
    "case, why",
    [
        ({"metrics_attempt": "a0"}, "latest render attempt"),
        ({"render_rc": 1}, "exited 1"),
        ({"clip": None}, "no clip"),
        ({"clip": "corrupt"}, "not usable video"),
    ],
)
def test_a_stale_or_failed_render_is_failed(tmp_path, case, why):
    results, scene = _collect(_run_dir(tmp_path, **case))
    assert scene["grade"] == "failed" and results["rendered"] == 0 and any(why in r for r in scene["reasons"])
    assert "video" not in scene["files"]


@needs_ffmpeg
def test_a_check_that_exits_nonzero_fails_whatever_it_printed(tmp_path):
    _, scene = _collect(_run_dir(tmp_path, check_rc=1))
    assert scene["check"]["result"] == "FAILED" and scene["grade"] == "unstable"


@needs_ffmpeg
def test_the_launcher_reports_failure_and_leaves_no_stale_metrics(tmp_path):
    run = _run_dir(tmp_path)
    env = dict(os.environ, PYTHON="/bin/false", EXAMPLES=str(tmp_path))
    proc = subprocess.run(["bash", str(SCRIPTS / "run_all.sh"), str(run), "0"], env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert (run / "failed.txt").read_text().split() == ["s"]
    assert not (run / "out" / "s" / "metrics.json").exists()
    _, scene = _collect(run)
    assert scene["grade"] == "failed"


def test_a_render_that_fails_inside_the_renderer_exits_nonzero(tmp_path, monkeypatch):
    pytest.importorskip("pyglet")
    monkeypatch.chdir(EXAMPLES)
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from common import cli, modes, presets

    render_scene = _module("render_scene")
    table = presets.read(cli.find("basic_pendulum"))["steel"]
    entry = {"scene": "basic_pendulum", "preset": "steel", "sim_seconds": 0.1, "speed": 1.0, "poster_frac": 0.5,
             "settings": {k: table[k] for k in ("sim", "collision", "solver", "model", "entries") if k in table}}  # fmt: skip
    (tmp_path / "plan.json").write_text(json.dumps({"render": [entry]}))

    def fail(*args, **kwargs):
        raise RuntimeError("injected build failure")

    monkeypatch.setattr(modes, "run_example", fail)
    monkeypatch.setattr(render_scene, "_patch_steel", lambda: None)
    argv = ["render_scene.py", "--plan", str(tmp_path / "plan.json"), "--scene", "basic_pendulum", "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        render_scene.main()
    assert exit_info.value.code == 1
    metrics = json.loads((tmp_path / "basic_pendulum" / "metrics.json").read_text())
    assert "injected build failure" in metrics["build_error"]
