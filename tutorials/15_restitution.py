# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Restitution: per-shape coefficients and the restitution_combine mode."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
BALL_E = [0.0, 0.3, 0.6, 0.9]


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(restitution=0.0))
    for i, e in enumerate(BALL_E):
        ball = builder.add_body(xform=wp.transform((0.3 * i - 0.45, 0.0, 1.06), wp.quat_identity()))
        builder.add_shape_sphere(
            ball, radius=0.06, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, restitution=e)
        )
    return builder.finalize()


def simulate(rec, combine):
    model = build()
    solver = SolverSteel(model, restitution_combine=combine)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(f'restitution_combine="{combine}"', model)
    for _ in range(rec.frames(4.0)):
        heights = state_0.body_q.numpy()[:, 2] - 0.06
        run.frame(state_0, **{f"e_{e}": h for e, h in zip(BALL_E, heights)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "average")  # default: the ground's 0 halves every coefficient
    simulate(rec, "max")
    rec.save()
