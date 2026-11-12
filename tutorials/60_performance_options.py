# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Performance options: parallel_tree, use_parallel_streams, double_buffer and reuse_kinematics."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
OPTIONS = {
    "defaults": {},
    "parallel_tree": {"parallel_tree": True},
    "use_parallel_streams + double_buffer": {"use_parallel_streams": True, "double_buffer": True},
    "reuse_kinematics": {"reuse_kinematics": True},
}


def tf(x, y, z, q=None):
    return wp.transform((x, y, z), q if q is not None else wp.quat_identity())


def build():
    """A trunk with eight driven three-link branches, plus a two-link arm as a second size group."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    trunk = builder.add_link(xform=tf(0.0, 0.0, 0.8))
    builder.add_shape_cylinder(trunk, radius=0.06, half_height=0.05)
    joints = [builder.add_joint_fixed(-1, trunk, parent_xform=tf(0.0, 0.0, 0.8))]
    for k in range(8):
        yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 2.0 * math.pi * k / 8)
        parent, anchor = trunk, wp.transform(wp.quat_rotate(yaw, wp.vec3(0.07, 0.0, 0.0)), yaw)
        for i in range(3):
            link = builder.add_link(xform=tf(0.0, 0.0, 0.8))
            builder.add_shape_capsule(
                link,
                xform=tf(
                    0.06, 0.0, 0.0, wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)
                ),
                radius=0.015,
                half_height=0.045,
                cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False),
            )
            axis = newton.Axis.Y if i else newton.Axis.Z
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    link,
                    axis=axis,
                    parent_xform=anchor,
                    target_ke=40.0,
                    target_kd=1.0,
                    armature=0.001,
                )
            )
            builder.joint_q[-1] = 0.3 * math.sin(k + i)
            parent, anchor = link, tf(0.12, 0.0, 0.0)
    builder.add_articulation(joints)
    base = builder.add_link(xform=tf(0.6, 0.0, 0.3))
    builder.add_shape_box(base, hx=0.02, hy=0.02, hz=0.1)
    j0 = builder.add_joint_revolute(
        -1, base, axis=newton.Axis.Y, parent_xform=tf(0.6, 0.0, 0.4), child_xform=tf(0.0, 0.0, 0.1)
    )
    tip = builder.add_link(xform=tf(0.6, 0.0, 0.1))
    builder.add_shape_box(tip, hx=0.02, hy=0.02, hz=0.1)
    j1 = builder.add_joint_revolute(
        base,
        tip,
        axis=newton.Axis.Y,
        parent_xform=tf(0.0, 0.0, -0.1),
        child_xform=tf(0.0, 0.0, 0.1),
    )
    builder.joint_q[-2:] = [0.8, 0.4]
    builder.add_articulation([j0, j1])
    return builder.finalize()


def simulate(rec, label, reference=None):
    model = build()
    solver = SolverSteel(model, **OPTIONS[label])
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    # sway the branch targets so the tree keeps moving
    targets = control.joint_target_q.numpy()

    run = rec.variant(label, model)
    for frame in range(rec.frames(4.0)):
        t = frame / rec.fps
        control.joint_target_q.assign((targets + 0.4 * math.sin(2.0 * t)).astype(np.float32))
        q = state_0.body_q.numpy()
        diff = (
            0.0
            if reference is None
            else 1000.0 * np.abs(q[:, :3] - reference.poses[frame][:, :3]).max()
        )
        run.frame(state_0, difference_mm=diff)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)
    print(f"{label}: reuse_kinematics runs = {solver.reuse_kinematics}")
    return run


if __name__ == "__main__":
    rec = Recorder()
    reference = simulate(rec, "defaults")
    for label in list(OPTIONS)[1:]:
        simulate(rec, label, reference)
    rec.save()
