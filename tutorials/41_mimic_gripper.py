# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""A parallel gripper whose fingers are coupled by a mimic joint grasps, lifts and shakes a box."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
SQUEEZE = 40.0  # N on the leader finger


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build(mu):
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
    finger_cfg = newton.ModelBuilder.ShapeConfig(density=800.0, mu=mu)
    base = builder.add_link(xform=tf(0.0, 0.0, 0.45))
    builder.add_shape_box(
        base, hx=0.16, hy=0.03, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=500.0)
    )
    lift = builder.add_joint_prismatic(
        -1,
        base,
        axis=newton.Axis.Z,
        parent_xform=tf(0.0, 0.0, 0.45),
        target_ke=4000.0,
        target_kd=200.0,
        armature=0.5,
    )
    left = builder.add_link(xform=tf(-0.12, 0.0, 0.3))
    builder.add_shape_box(left, hx=0.01, hy=0.03, hz=0.12, cfg=finger_cfg)
    leader = builder.add_joint_prismatic(
        base,
        left,
        axis=newton.Axis.X,
        parent_xform=tf(-0.12, 0.0, -0.3),
        armature=0.05,
        limit_lower=0.0,
        limit_upper=0.12,
    )
    right = builder.add_link(xform=tf(0.12, 0.0, 0.3))
    builder.add_shape_box(right, hx=0.01, hy=0.03, hz=0.12, cfg=finger_cfg)
    follower = builder.add_joint_prismatic(
        base,
        right,
        axis=newton.Axis.X,
        parent_xform=tf(0.12, 0.0, -0.3),
        armature=0.05,
        limit_lower=-0.12,
        limit_upper=0.0,
    )
    builder.add_articulation([lift, leader, follower])
    builder.set_joint_mimic(follower, leader, coeffs=(0.0, -1.0))  # q_right = -q_left
    box = builder.add_body(xform=tf(0.0, 0.0, 0.0601))
    builder.add_shape_box(
        box, hx=0.05, hy=0.05, hz=0.06, cfg=newton.ModelBuilder.ShapeConfig(density=600.0, mu=mu)
    )
    return builder.finalize(), base, box


def simulate(rec, label, mu, **options):
    model, base, box = build(mu)
    # mimic rows and finger contacts share the articulation's row capacity
    solver = SolverSteel(
        model, enable_joint_limits=True, articulation_constraint_max=128, **options
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for frame in range(rec.frames(7.0)):
        t = frame / rec.fps
        q = state_0.body_q.numpy()
        run.frame(state_0, slip_mm=1000.0 * ((q[base, 2] - q[box, 2]) - (0.45 - 0.0601)))
        lift = min(0.25, max(0.0, 0.25 * (t - 1.0)))
        shake = 0.03 * math.sin(2.0 * math.pi * 3.0 * (t - 2.5)) if 2.5 < t < 5.5 else 0.0
        control.joint_target_q.assign(np.array([lift + shake, 0.0, 0.0], dtype=np.float32))
        control.joint_f.assign(np.array([0.0, SQUEEZE if t > 0.3 else -5.0, 0.0], dtype=np.float32))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "mu 0.6, patch friction (default)", 0.6)
    simulate(rec, "mu 0.6, point friction", 0.6, friction_anchor_beta=0.0)
    simulate(rec, "mu 0.15, patch friction", 0.15)
    rec.save()
