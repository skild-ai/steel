# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Dry joint friction (MuJoCo frictionloss) with enable_joint_friction."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
FRICTION = [0.0, 0.3, 1.0]  # N m


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    builder = newton.ModelBuilder()
    for i, friction in enumerate(FRICTION):
        y = 0.45 * i - 0.45
        link = builder.add_link(xform=tf(0.0, y, 0.8))
        builder.add_shape_capsule(
            link,
            xform=tf(0.0, 0.0, -0.2),
            radius=0.02,
            half_height=0.18,
            cfg=newton.ModelBuilder.ShapeConfig(density=400.0),
        )
        builder.add_shape_sphere(
            link,
            xform=tf(0.0, 0.0, -0.4),
            radius=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(density=2000.0),
        )
        j = builder.add_joint_revolute(
            -1, link, axis=newton.Axis.Y, parent_xform=tf(0.0, y, 0.8), friction=friction
        )
        builder.add_articulation([j])
    model = builder.finalize()
    model.joint_q.fill_(math.radians(70.0))
    return model


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(5.0)):
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, **{f"f_{f}": q[i] for i, f in enumerate(FRICTION)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "enable_joint_friction=False (default)")
    simulate(rec, "enable_joint_friction=True", enable_joint_friction=True)
    rec.save()
