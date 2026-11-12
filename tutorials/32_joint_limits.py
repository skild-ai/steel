# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Joint position limits are off by default; enable_joint_limits turns them into unilateral rows."""

import math

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    builder = newton.ModelBuilder()
    link = builder.add_link(xform=tf(0.0, 0.0, 0.8))
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
    limit = math.radians(30.0)
    j = builder.add_joint_revolute(
        -1,
        link,
        axis=newton.Axis.Y,
        parent_xform=tf(0.0, 0.0, 0.8),
        limit_lower=-limit,
        limit_upper=limit,
    )
    builder.add_articulation([j])
    model = builder.finalize()
    model.joint_q.fill_(math.radians(80.0))  # released outside the limit
    return model


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        run.frame(state_0, angle_deg=math.degrees(state_0.joint_q.numpy()[0]))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "enable_joint_limits=False (default)")
    simulate(rec, "enable_joint_limits=True", enable_joint_limits=True)
    rec.save()
