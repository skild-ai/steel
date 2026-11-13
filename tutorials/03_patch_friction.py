# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Patch friction vs point friction: a box at 98% of its friction limit on a slope."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8  # 30 Hz frames
MU = 0.5
SLOPE = math.atan(0.49)  # tan(slope) = 0.98 * MU


def build():
    builder = newton.ModelBuilder()
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -SLOPE)
    builder.add_shape_box(
        -1,
        xform=wp.transform((0.0, 0.0, 0.3), tilt),
        hx=0.8,
        hy=0.3,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(mu=MU),
    )
    normal = np.array(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, 1.0)))
    start = np.array([0.0, 0.0, 0.3]) + normal * 0.11
    box = builder.add_body(xform=wp.transform(start, tilt))
    builder.add_shape_box(
        box, hx=0.12, hy=0.08, hz=0.06, cfg=newton.ModelBuilder.ShapeConfig(mu=MU)
    )
    return builder.finalize(), start


def simulate(rec, label, **options):
    model, start = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(10.0)):
        slip = np.linalg.norm(state_0.body_q.numpy()[0, :3] - start)
        run.frame(state_0, slip_mm=1000.0 * slip)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "point friction (friction_anchor_beta=0)", friction_anchor_beta=0.0)
    simulate(rec, "patch friction (default)")
    rec.save()
