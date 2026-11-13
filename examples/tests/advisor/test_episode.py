# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Episode probe on real Newton models (CPU): start-state check, mass-ratio islands, capacities."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from common.advisor.episode import EpisodeProbe, model_facts, solver_capacities, start_state_facts  # noqa: E402
from common.advisor.stability import recommend_stability  # noqa: E402

DT = 1.0 / 240.0


def _free_box_model(mass_scale=1.0):
    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), label="box")
    b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0 * mass_scale))
    b.add_ground_plane()
    return b.finalize(device="cpu")


def test_start_state_from_fk_is_consistent():
    model = _free_box_model()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    st = start_state_facts(model, state)
    assert st["checked"] and st["pose_mismatch_bodies"] == 0 and st["velocity_mismatch_bodies"] == 0


def test_velocity_written_only_to_body_qd_is_caught():
    model = _free_box_model()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    qd = state.body_qd.numpy()
    qd[0, :3] = (2.0, 0.0, 0.0)
    state.body_qd.assign(qd)
    st = start_state_facts(model, state)
    assert st["velocity_mismatch_bodies"] == 1 and st["velocity_in_body_qd_only_bodies"] == 1
    advice = recommend_stability({"start_state": st, "episode": {"row_watermarks": {}}, "settings": {}})
    assert any("joint_q / joint_qd" in m for lv, m in advice["flags"] if lv == "critical")


def test_pose_written_only_to_body_q_is_caught():
    model = _free_box_model()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    q = state.body_q.numpy()
    q[0, 2] += 0.5
    state.body_q.assign(q)
    st = start_state_facts(model, state)
    assert st["pose_mismatch_bodies"] == 1 and st["pose_mismatch_max_m"] == pytest.approx(0.5, abs=1e-5)


def _stack(ratio):
    """A heavy box resting on a light one on the ground; collision run once."""
    b = newton.ModelBuilder()
    light = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), label="light")
    b.add_shape_box(light, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=100.0))
    heavy = b.add_body(xform=wp.transform((0.0, 0.0, 0.299), wp.quat_identity()), label="heavy")
    b.add_shape_box(heavy, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=100.0 * ratio))
    b.add_ground_plane()
    model = b.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return model, state, contacts


def test_island_mass_ratio_of_a_heavy_on_light_stack():
    model, state, contacts = _stack(50.0)
    probe = EpisodeProbe(model, None, DT, DT, 0.0005)
    probe.update(state, contacts, frame_ms=1.0)
    assert probe.island_ratio_max == pytest.approx(50.0, rel=1e-4)
    assert probe.pair_ratio_max == pytest.approx(50.0, rel=1e-4)


def test_model_facts_count_limited_dofs():
    b = newton.ModelBuilder()
    link = b.add_link(label="link")
    b.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    j = b.add_joint_revolute(-1, link, axis=(0.0, 0.0, 1.0), limit_lower=-1.0, limit_upper=1.0)
    b.add_articulation([j])
    model = b.finalize(device="cpu")
    facts = model_facts(model, None)
    assert facts["limited_dofs_world0"] == 1 and facts["articulations_world0"] == 1


def test_solver_capacities_resolve_the_propagation_default():
    class Fake:
        articulation_constraint_max = 32
        free_body_constraint_max = 512

    assert solver_capacities(Fake()) == {"articulation": 32, "free_body": 512, "propagation": 544}


def test_episode_probe_end_to_end_with_steel_on_cpu():
    model = _free_box_model()
    solver = newton.solvers.SolverSteel(model, row_watermark=True)
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    probe = EpisodeProbe(model, solver, DT, DT, None)
    for _ in range(20):
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0
        probe.update(s0, contacts, frame_ms=1.0)
    out = probe.finish()
    assert out["capacity_check"] == "ok" and out["frames"] == 20
    assert isinstance(out["row_watermarks"], dict) and out["row_watermarks"]["free_body_overflow_world_steps"] == 0
    assert out["capacities"]["articulation"] == 32 and np.isfinite(out["penetration_peak_m"])


def test_a_body_entering_a_static_box_is_counted_once():
    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="box")
    b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), hx=1.0, hy=1.0, hz=0.2, label="deck")
    model = b.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    probe = EpisodeProbe(model, None, DT, DT, None)
    probe.update(state, None)
    assert not probe.sank
    q = state.body_q.numpy()
    q[0, 2] = 0.1  # centre 0.1 m deep in the 0.2 m half-height deck
    state.body_q.assign(q)
    probe.update(state, None)
    probe.update(state, None)
    assert probe.sank == {0} and probe.sank_depth_max == pytest.approx(0.1, abs=1e-4)


def test_a_body_inside_a_box_after_the_first_step_counts_when_the_start_is_observed():
    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label="box")
    b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), hx=1.0, hy=1.0, hz=0.2, label="deck")
    model = b.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    probe = EpisodeProbe(model, None, DT, DT, None)
    probe.observe_initial(state)  # the runner's order: observe the start, then step, then update
    q = state.body_q.numpy()
    q[0, 2] = 0.1  # the first step lands the centre inside the deck
    state.body_q.assign(q)
    probe.update(state, None)
    assert probe.sank == {0}
    assert probe.finish()["unchanged_frames"] == 0


def test_frozen_counts_repeats_from_the_start_but_not_late_settling():
    model = _free_box_model()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    frozen = EpisodeProbe(model, None, DT, DT, None)
    frozen.observe_initial(state)
    for _ in range(5):
        frozen.update(state, None)  # the exposed state never changes
    assert frozen.unchanged_frames == 4
    settling = EpisodeProbe(model, None, DT, DT, None)
    q = state.body_q.numpy()
    for k in range(5):
        q[0, 2] = 0.1 + 0.01 * min(k, 2)  # moves for two frames, then rests exactly
        state.body_q.assign(q)
        settling.update(state, None)
    assert settling.unchanged_frames == 0
