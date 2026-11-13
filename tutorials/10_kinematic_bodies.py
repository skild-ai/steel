# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Kinematic bodies and per-body gravity: a prescribed paddle sweeps boxes."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
SPIN = 1.2  # rad/s


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    paddle = builder.add_body(
        xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()), is_kinematic=True
    )
    builder.add_shape_box(paddle, hx=0.42, hy=0.015, hz=0.04)
    for k in range(10):
        a = 2.0 * math.pi * (k + 0.5) / 10
        box = builder.add_body(
            xform=wp.transform((0.32 * math.cos(a), 0.32 * math.sin(a), 0.03), wp.quat_identity())
        )
        builder.add_shape_box(
            box,
            hx=0.03,
            hy=0.03,
            hz=0.03,
            cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.5),
        )
    floater = builder.add_body(
        xform=wp.transform((0.0, 0.0, 0.4), wp.quat_identity()), disable_gravity=True
    )
    builder.add_shape_box(
        floater, hx=0.04, hy=0.04, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=300.0)
    )
    model = builder.finalize()
    qd = model.joint_qd.numpy()
    qd[-3:] = (0.3, 0.5, 0.0)  # the floater tumbles in place
    model.joint_qd.assign(qd)
    return model


def move_paddle(model, state, t):
    """Prescribe the paddle's pose at time t; Steel holds kinematic joint coordinates as given."""
    angle = SPIN * t
    q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
    q[0:7] = (0.0, 0.0, 0.05, 0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
    qd[0:6] = (0.0, 0.0, 0.0, 0.0, 0.0, SPIN)
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    # collision reads body poses, so bring them up to date with the joint coordinates
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    t = 0.0
    for _ in range(rec.frames(5.0)):
        q = state_0.body_q.numpy()
        box_r = np.linalg.norm(q[1:-1, :2], axis=1)
        run.frame(state_0, mean_box_radius=box_r.mean(), floater_z=q[-1, 2])
        for _ in range(SUBSTEPS):
            move_paddle(model, state_0, t + DT)
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            t += DT
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "kinematic paddle at 1.2 rad/s")
    rec.save()
