# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Friction-iteration options: fewer friction rows with the same result, and where that breaks."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
MU = 0.5


def build_slope():
    """Three boxes on a slope at 90% of their friction limit."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -math.atan(0.9 * MU))
    cfg = newton.ModelBuilder.ShapeConfig(mu=MU)
    builder.add_shape_box(
        -1, xform=wp.transform((0.0, 0.0, 0.3), tilt), hx=0.6, hy=0.3, hz=0.05, cfg=cfg
    )
    normal = np.array(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, 1.0)))
    for y in (-0.15, 0.0, 0.15):
        box = builder.add_body(xform=wp.transform(np.array([0.0, y, 0.3]) + 0.1 * normal, tilt))
        builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    return builder.finalize()


def build_pile():
    """Twenty-seven boxes with a 3 cm contact gap, so many reported contacts are still apart."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.6, gap=0.03)
    builder.add_ground_plane(cfg=cfg)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pose = wp.transform(
                    (0.13 * i - 0.13, 0.13 * j - 0.13, 0.06 + 0.13 * k),
                    wp.quat_rpy(0.2 * i, 0.1 * j, 0.3 * k),
                )
                box = builder.add_body(xform=pose)
                builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    return builder.finalize()


def simulate_slope(rec, label, **options):
    model = build_slope()
    solver = SolverSteel(model, friction_anchor_beta=0.0, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    start = state_0.body_q.numpy()[:, :3].copy()

    run = rec.variant(label, model, scene="slope")
    for _ in range(rec.frames(3.0)):
        slip = np.linalg.norm(state_0.body_q.numpy()[:, :3] - start, axis=1).max()
        run.frame(state_0, slip_mm=1000.0 * slip)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


def simulate_pile(rec, label, **options):
    model = build_pile()
    solver = SolverSteel(model, row_watermark=True, free_body_constraint_max=2048, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model, scene="pile")
    for _ in range(rec.frames(3.0)):
        rows = solver.constraint_row_watermarks()["free_body_raw_high_water"]
        run.frame(state_0, top_z=state_0.body_q.numpy()[:, 2].max(), rows_high_water=rows)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    # friction rows in every iteration (default), only the last 4 or 1, or never
    for k in (-1, 4, 1, 0):
        simulate_slope(
            rec, f"contact_friction_position_iterations={k}", contact_friction_position_iterations=k
        )
    simulate_pile(rec, "pile, every contact gets friction rows (default)")
    simulate_pile(rec, "pile, contact_friction_gap_max=0", contact_friction_gap_max=0.0)
    rec.save()
