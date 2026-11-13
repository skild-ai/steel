# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Acceptance end to end: what may never count as a healthy trial, and the trial process lifecycle."""

import json
import sys
import textwrap
import time
from argparse import Namespace

import numpy as np
import pytest

from common.advisor import loop
from common.advisor.stability import healthy, no_worse, recommend_stability, summarize

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from common.advisor.episode import EpisodeProbe, model_facts  # noqa: E402
from common.advisor.health import health_flags  # noqa: E402


def _facts(**health):
    return {
        "settings": {"sim": {"substeps": 4}, "solver": {}},
        "solver_kwargs": {"iterations": 48, "friction_anchor_beta": 0.2},
        "frames_planned": 2,
        "frames_done": 2,
        "frame_dt": 1 / 60,
        "episode": {
            "frames": 2,
            "row_watermarks": {},
            "capacity_check": "ok",
            "health": {"nonfinite_frames": 0, **health},
            "frame_ms_median": 10.0,
        },
        "check": {"ok": True},
    }


def _result(facts, attempt=None):
    return {
        "status": "ok",
        "settings": facts["settings"],
        "facts": facts,
        "advice": recommend_stability(facts),
        "summary": summarize(facts) | {"check_ok": True},
        "attempt": attempt,
    }


def _box_on_ground():
    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    b.add_ground_plane()
    model = b.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state


def test_invalid_contact_geometry_is_never_healthy():
    model, state = _box_on_ground()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    assert contacts.rigid_contact_count.numpy()[0] > 0
    normals = contacts.rigid_contact_normal.numpy()
    normals[0] = np.nan
    contacts.rigid_contact_normal.assign(normals)
    probe = EpisodeProbe(model, None, 1 / 240, 1 / 240, None)
    probe.update(state, contacts, 10.0)
    facts = _facts()
    facts["episode"] = probe.finish() | {"capacity_check": "ok", "row_watermarks": {}}
    facts["frames_done"] = facts["frames_planned"] = 1
    assert facts["episode"]["health"]["invalid_contact_frames"] == 1
    assert any(level == "critical" for level, _ in health_flags(facts["episode"]["health"], {}))
    result = _result(facts)
    assert any("non-finite geometry" in m for m in result["advice"]["critical"])
    assert result["summary"]["invalid_contact_frames"] == 1
    assert not healthy(result)


def test_rotational_slip_cannot_be_traded_for_speed():
    hold = {"robot": "finger", "object": "box", "object_free": True, "frames": 60, "slip_mm": 1.0, "slip_deg": 1.0}
    baseline = _result(_facts(grasp_holds=[hold]))
    facts = _facts(grasp_holds=[hold | {"slip_deg": 170.0}])
    facts["episode"]["frame_ms_median"] = 5.0
    candidate = _result(facts)
    assert healthy(baseline)
    assert not no_worse(baseline["summary"], candidate["summary"])
    # At the iteration ceiling no setting is left to try; the slip stays reported.
    assert any("Unresolved grasp slip" in m for lv, m in candidate["advice"]["flags"] if lv == "warn")


def test_a_measured_health_number_that_disappears_is_unknown():
    base = {"frame_ms": 10.0, "grasp_slip_max_mm": 1.0, "grasp_slip_max_deg": 1.0}
    assert not no_worse(base, {"frame_ms": 5.0, "grasp_slip_max_mm": None, "grasp_slip_max_deg": None})
    assert no_worse(base, {"frame_ms": 5.0, "grasp_slip_max_mm": 2.0, "grasp_slip_max_deg": 3.0})


def _tune_args(tmp_path, **over):
    return Namespace(
        example="x",
        start="preset",
        max_trials=1,
        cost_trials=0,
        seconds=None,
        no_check=True,
        timeout=1,
        out=str(tmp_path),
        **over,
    )


def _stub_shelf(monkeypatch, tmp_path):
    monkeypatch.setattr(loop.cli, "find", lambda name: tmp_path / name)
    monkeypatch.setattr(loop.presets, "read", lambda path: {"steel": {"sim": {"substeps": 4}, "solver": {}}})


@pytest.mark.parametrize("rc", [17, "timeout", 0])
def test_a_result_left_by_an_earlier_run_is_never_reused(tmp_path, monkeypatch, rc):
    _stub_shelf(monkeypatch, tmp_path)
    (tmp_path / "trial0").mkdir()
    stale = _result(_facts())
    stale["settings"] = {"sim": {"substeps": 99}, "solver": {}}
    (tmp_path / "trial0" / "trial.json").write_text(json.dumps(stale))
    monkeypatch.setattr(loop, "_spawn", lambda *a: rc)  # the child writes nothing
    loop.run_tune(_tune_args(tmp_path), [])
    final = json.loads((tmp_path / "tune.json").read_text())
    assert not final["final_healthy"] and final["trials"][0]["status"] == "process_failed"


def test_a_failed_process_fails_its_trial_even_after_writing_a_healthy_result(tmp_path, monkeypatch):
    _stub_shelf(monkeypatch, tmp_path)

    def child(cmd, log, timeout):
        out = cmd[cmd.index("--out") + 1]
        res = _result(_facts(), attempt=cmd[cmd.index("--attempt") + 1])
        (loop.Path(out) / "trial.json").write_text(json.dumps(res))
        return 1

    monkeypatch.setattr(loop, "_spawn", child)
    loop.run_tune(_tune_args(tmp_path), [])
    final = json.loads((tmp_path / "tune.json").read_text())
    assert not final["final_healthy"] and final["trials"][0]["status"] == "process_failed"


def test_an_exception_after_the_episode_fails_the_trial(tmp_path, monkeypatch):
    _stub_shelf(monkeypatch, tmp_path)
    monkeypatch.setattr(loop.presets, "select", lambda *a, **k: None)
    monkeypatch.setattr(loop, "steel_params", lambda: {})

    def run(path, argv, episode):
        episode.ran = True
        episode.result = _result(_facts())
        raise RuntimeError("example failed after run()")

    monkeypatch.setattr(loop.modes, "run_example", run)
    args = Namespace(example="x", settings=None, out=str(tmp_path), frames=2, seconds=None, no_check=True, timeout=1)
    loop.run_trial(args, [])
    trial = json.loads((tmp_path / "trial.json").read_text())
    assert trial["status"] == "error" and trial["error"] == "RuntimeError: example failed after run()"
    assert any("raised after its episode" in m for m in trial["advice"]["critical"])
    assert not healthy(trial)


def test_an_episode_that_raised_before_its_result_keeps_structured_facts(tmp_path, monkeypatch):
    _stub_shelf(monkeypatch, tmp_path)
    monkeypatch.setattr(loop.presets, "select", lambda *a, **k: None)
    monkeypatch.setattr(loop, "steel_params", lambda: {"defaults": {}})

    def run(path, argv, episode):
        episode.ran = True
        raise ValueError("probe setup failed")

    monkeypatch.setattr(loop.modes, "run_example", run)
    args = Namespace(example="x", settings=None, out=str(tmp_path), frames=2, seconds=None, no_check=True, timeout=1)
    loop.run_trial(args, [])
    trial = json.loads((tmp_path / "trial.json").read_text())
    assert trial["status"] == "error" and trial["facts"]["build_error"] == "ValueError: probe setup failed"
    assert not healthy(trial)


def test_the_solver_snapshot_records_sleeping():
    class Solver:
        enable_sleeping = True

    assert loop._solver_kwargs(Solver())["enable_sleeping"] is True


def test_d6_limit_violations_are_measured():
    b = newton.ModelBuilder()
    body = b.add_link(label="slider")
    b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    axis = newton.ModelBuilder.JointDofConfig
    joint = b.add_joint_d6(
        -1,
        body,
        linear_axes=[axis(axis=(1.0, 0.0, 0.0), limit_lower=-1.0, limit_upper=1.0)],
        angular_axes=[axis(axis=(0.0, 0.0, 1.0), limit_lower=-0.5, limit_upper=0.5)],
    )
    b.add_articulation([joint])
    model = b.finalize(device="cpu")
    state = model.state()
    for q, excess in (([2.0, 0.0], 1.0), ([0.0, -0.8], 0.3)):
        state.joint_q.assign(np.array(q, dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        assert model_facts(model, None)["limited_dofs_world0"] == 2
        probe = EpisodeProbe(model, None, 1 / 240, 1 / 240, None)
        probe.update(state, None, 10.0)
        assert probe.finish()["joint_limit_excess_max"] == pytest.approx(excess, abs=1e-5)


@pytest.mark.skipif(sys.platform == "win32", reason="process groups")
def test_a_timed_out_trial_ends_its_whole_process_group(tmp_path, monkeypatch):
    marker = tmp_path / "descendant-ran"
    script = textwrap.dedent(
        f"""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", "import time, pathlib; time.sleep(2); pathlib.Path({str(marker)!r}).touch()"])
        time.sleep(30)
        """
    )
    monkeypatch.setattr(loop.cli, "ROOT", tmp_path)
    t = time.perf_counter()
    rc = loop._spawn([sys.executable, "-c", script], tmp_path / "trial.log", timeout=1)
    assert rc == "timeout" and time.perf_counter() - t < 15
    time.sleep(3)
    assert not marker.exists()


def test_a_finished_trial_leaves_no_descendant_behind(tmp_path, monkeypatch):
    marker = tmp_path / "orphan-ran"
    script = textwrap.dedent(
        f"""
        import subprocess, sys
        subprocess.Popen([sys.executable, "-c", "import time, pathlib; time.sleep(2); pathlib.Path({str(marker)!r}).touch()"])
        """
    )
    monkeypatch.setattr(loop.cli, "ROOT", tmp_path)
    assert loop._spawn([sys.executable, "-c", script], tmp_path / "trial.log", timeout=30) == 0
    time.sleep(3)
    assert not marker.exists()


def test_spawn_returns_the_childs_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(loop.cli, "ROOT", tmp_path)
    assert loop._spawn([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "t.log", timeout=30) == 3
