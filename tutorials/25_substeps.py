# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Substeps vs iterations: split the same 30 Hz frame into more, cheaper solver steps."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

FRAME_DT = 1.0 / 30.0
N, H = 10, 0.04


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    for i in range(N):
        box = builder.add_body(
            xform=wp.transform((0.0, 0.0, H + 2 * H * i + 0.0002 * i), wp.quat_identity())
        )
        builder.add_shape_box(box, hx=H, hy=H, hz=H, cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    return builder.finalize()


def simulate(rec, substeps, iterations):
    model = build()
    solver = SolverSteel(model, iterations=iterations)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    dt = FRAME_DT / substeps

    run = rec.variant(f"{substeps} substeps x {iterations} iterations", model)
    for _ in range(rec.frames(4.0)):
        top = state_0.body_q.numpy()[N - 1, 2]
        run.frame(state_0, sag_mm=1000.0 * (H * (2 * N - 1) - top))
        for _ in range(substeps):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, 2, 24)  # 48 sweeps per frame
    simulate(rec, 4, 12)
    simulate(rec, 8, 6)
    simulate(rec, 16, 3)
    rec.save()
