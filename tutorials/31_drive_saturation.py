# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Drive saturation: drive_mode="augmented" vs "rows" when a stiff drive hits its effort limit."""

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
    arm = builder.add_link(xform=tf(0.25, 0.0, 0.6))
    builder.add_shape_box(
        arm, hx=0.25, hy=0.02, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=200.0)
    )
    builder.add_shape_sphere(
        arm,
        xform=tf(0.25, 0.0, 0.0),
        radius=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(density=3000.0),
    )
    j = builder.add_joint_revolute(
        -1,
        arm,
        axis=(0.0, -1.0, 0.0),
        parent_xform=tf(0.0, 0.0, 0.6),
        child_xform=tf(-0.25, 0.0, 0.0),
        target_ke=3000.0,
        target_kd=60.0,
        target_q=0.0,
        effort_limit=12.0,
    )
    builder.add_articulation([j])
    model = builder.finalize()
    model.joint_q.fill_(math.radians(-60.0))  # released 60 degrees below the target
    return model


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(3.0)):
        run.frame(state_0, angle_deg=math.degrees(state_0.joint_q.numpy()[0]))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, 'drive_mode="augmented" (default)')
    simulate(rec, 'drive_mode="rows"', drive_mode="rows")
    rec.save()
