# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""HealthProbe on real Newton models, states and contacts (CPU), and health_flags on its facts."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from common.advisor.health import CLASSES, ROLE_OBJECT, ROLE_SUPPORT, HealthProbe, health_flags  # noqa: E402

DT = 1.0 / 240.0


def _probe(model):
    return HealthProbe(model, None, DT, DT, 0.0005)


def _collide(model, state):
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return contacts


def _box_on_ground():
    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), label="box")
    b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    b.add_ground_plane()
    model = b.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state


def test_finite_resting_box_is_clean():
    model, state = _box_on_ground()
    probe = _probe(model)
    for _ in range(3):
        probe.update(state, _collide(model, state))
    h = probe.finish()
    assert h["nonfinite_frames"] == 0
    assert h["penetration_max_by_class"]["object_support"] < 1e-3
    assert not [lvl for lvl, _ in health_flags(h, {}) if lvl in ("warn", "critical")]


@pytest.mark.parametrize("field, index", [("body_q", 0), ("body_q", 4), ("body_qd", 4)])
def test_nonfinite_state_is_critical(field, index):
    model, state = _box_on_ground()
    probe = _probe(model)
    probe.update(state, _collide(model, state))
    values = getattr(state, field).numpy()
    values[0, index] = np.nan  # position x, a quaternion component, or angular velocity x
    getattr(state, field).assign(values)
    probe.update(state, None)
    h = probe.finish()
    assert h["nonfinite_frames"] == 1 and h["nonfinite_first"]["frame"] == 2
    flags = health_flags(h, {})
    assert any(lvl == "critical" and "Non-finite" in msg for lvl, msg in flags)


@pytest.mark.parametrize("support_first", [True, False])
def test_object_side_of_a_fixed_support_is_chosen_by_role(support_first):
    """A free box pressed into a zero-DOF fixed-root slab: the box, not the slab, is the penetrating object."""
    b = newton.ModelBuilder()

    def add_support():
        link = b.add_link(label="slab")
        b.add_articulation([b.add_joint_fixed(-1, link)])
        b.add_shape_box(link, hx=1.0, hy=1.0, hz=0.1)

    def add_object():
        body = b.add_body(xform=wp.transform((0.0, 0.0, 0.1 + 0.2 - 0.35), wp.quat_identity()), label="cube")
        b.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)

    for add in (add_support, add_object) if support_first else (add_object, add_support):
        add()
    model = b.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    probe = _probe(model)
    probe.update(state, _collide(model, state))
    h = probe.finish()
    assert h["object_support_penetration_body"] == "cube"
    # The narrow phase resolves the 0.2 m overlap through the slab's bottom face: 0.25 m deep, the cube's
    # 0.2 m half-extent as denominator, in either shape order.
    assert h["object_support_penetration_over_half_thickness"] == pytest.approx(0.25 / 0.2, rel=0.05)
    assert any(lvl == "warn" and "cube" in msg for lvl, msg in health_flags(h, {}))


def test_passive_compound_object_is_an_object_on_its_support():
    """An unactuated FREE + FIXED pair is an object, and its contact with the ground is an object-support pair."""
    b = newton.ModelBuilder()
    root = b.add_link(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    child = b.add_link(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
    j0 = b.add_joint_free(root)
    j1 = b.add_joint_fixed(root, child, parent_xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()))
    b.add_articulation([j0, j1])
    b.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
    b.add_shape_box(child, hx=0.1, hy=0.1, hz=0.1)
    b.add_ground_plane()
    model = b.finalize(device="cpu")
    probe = _probe(model)
    assert probe.role[root] == ROLE_OBJECT and probe.role[child] == ROLE_OBJECT
    assert probe.role[-1] == ROLE_SUPPORT
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    probe.update(state, _collide(model, state))
    assert "object_support" in probe.finish()["penetration_max_by_class"]
    assert CLASSES.index("object_support") >= 0


def _finger_and_box(on_table: bool):
    """A driven prismatic finger touching a free box; the box rests on a table or hangs free."""
    b = newton.ModelBuilder()
    finger = b.add_link(label="finger")
    dof = newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), target_ke=1000.0, target_kd=10.0)
    b.add_articulation(
        [b.add_joint_prismatic(-1, finger, parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), axis=dof)]
    )
    b.add_shape_box(finger, hx=0.05, hy=0.05, hz=0.05)
    box = b.add_body(xform=wp.transform((0.0999, 0.0, 0.5), wp.quat_identity()), label="box")
    b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    if on_table:
        table = b.add_shape_box(-1, xform=wp.transform((0.1, 0.0, 0.4), wp.quat_identity()), hx=0.2, hy=0.2, hz=0.0501)
        del table
    model = b.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state, box


@pytest.mark.parametrize("on_table", [False, True])
def test_grasp_slip_counts_only_while_carried(on_table):
    model, state, box = _finger_and_box(on_table)
    probe = _probe(model)
    q = state.body_q.numpy()
    for f in range(40):
        q[box, 1] = 0.0005 * f  # the box slides 0.5 mm per frame along the finger face, 19.5 mm in all
        state.body_q.assign(q)
        probe.update(state, _collide(model, state))
    holds = probe.finish()["grasp_holds"]
    assert holds and holds[0]["object"] == "box" and holds[0]["robot"] == "finger"
    if on_table:
        assert holds[0]["slip_mm"] < 1.0
    else:
        assert holds[0]["slip_mm"] == pytest.approx(19.5, abs=0.5)
        assert any("Grasp slip" in msg for _, msg in health_flags(probe.finish(), {}))


def test_warm_start_flag_is_a_cohort_heuristic():
    h = {
        "frames": 10,
        "nonfinite_frames": 0,
        "match_threshold_m": 0.0005,
        "warmstart_match_fast": {"robot_object": {"matched_frac": 0.2, "contacts": 100}},
    }
    msg = " ".join(m for _, m in health_flags(h, {"contact_warmstart": True}))
    assert "cohort" in msg and "never" not in msg
    assert not [m for _, m in health_flags(h, {"contact_warmstart": False}) if "cohort" in m]


@pytest.mark.parametrize("field", ["rigid_contact_point0", "rigid_contact_normal", "rigid_contact_margin0"])
def test_nonfinite_contact_geometry_is_critical(field):
    """Non-finite contact geometry with a finite body state is reported, never read as zero penetration."""
    model, state = _box_on_ground()
    contacts = _collide(model, state)
    n = int(contacts.rigid_contact_count.numpy()[0])
    assert n > 0
    array = getattr(contacts, field)
    values = array.numpy()
    values[0] = np.nan  # one active contact; unused capacity stays untouched
    array.assign(values)
    probe = _probe(model)
    probe.update(state, contacts)
    h = probe.finish()
    assert h["nonfinite_frames"] == 0
    assert h["invalid_contact_frames"] == 1 and h["invalid_contacts"] == 1
    assert h["invalid_contact_first"]["body"] == "box"
    assert any(lvl == "critical" and "non-finite geometry" in msg for lvl, msg in health_flags(h, {}))
