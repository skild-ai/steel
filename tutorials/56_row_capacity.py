# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Row capacities: measure with row_watermark, catch overflow with check_constraint_capacity."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build():
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
    builder.add_ground_plane(cfg=cfg)
    for i in range(3):
        for j in range(3):
            box = builder.add_body(
                xform=wp.transform(
                    (0.12 * i - 0.12, 0.12 * j - 0.12, 0.3 + 0.03 * ((i + j) % 3)),
                    wp.quat_rpy(0.1 * i, 0.1 * j, 0.0),
                )
            )
            builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    return builder.finalize()


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(model, row_watermark=True, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(2.5)):
        marks = solver.constraint_row_watermarks()
        run.frame(
            state_0,
            rows_high_water=marks["free_body_raw_high_water"],
            lowest_z=state_0.body_q.numpy()[:, 2].min(),
        )
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)
    try:
        solver.check_constraint_capacity()
        print(f"{label}: fits, peak {marks['free_body_raw_high_water']} free-body rows")
    except RuntimeError as error:
        print(f"{label}: {error}")


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "free_body_constraint_max=512 (default)")
    simulate(rec, "free_body_constraint_max=200: rows dropped", free_body_constraint_max=200)
    rec.save()
