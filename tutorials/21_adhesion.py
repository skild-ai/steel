# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Sticky contacts with steel:adhesion: a box hangs under a ceiling until it is pulled off."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
HALF, GAP = 0.05, 0.004
WEIGHT = 1000.0 * (2 * HALF) ** 3 * 9.81  # 9.81 N


def build(factor):
    builder = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, gap=GAP)
    builder.add_ground_plane()
    ceiling = builder.add_shape_box(
        -1,
        xform=wp.transform((0.0, 0.0, 1.05), wp.quat_identity()),
        hx=0.2,
        hy=0.2,
        hz=0.05,
        cfg=cfg,
    )
    box = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0 - HALF), wp.quat_identity()))
    builder.add_shape_box(box, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
    model = builder.finalize()
    adhesion = model.steel.adhesion.numpy()
    adhesion[ceiling] = factor * WEIGHT / 4.0  # [N] per contact point; the face has four
    model.steel.adhesion.assign(adhesion)
    return model


def simulate(rec, label, factor, pull_at=None):
    model = build(factor)
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for frame in range(rec.frames(2.5)):
        t = frame / rec.fps
        run.frame(state_0, drop_mm=1000.0 * ((1.0 - HALF) - state_0.body_q.numpy()[0, 2]))
        for _ in range(SUBSTEPS):
            state_0.clear_forces()
            if pull_at is not None and t > pull_at:
                pull = 2.5 * WEIGHT * min(1.0, t - pull_at)  # ramps to 2.5 x weight over 1 s
                state_0.body_f.assign(np.array([[0, 0, -pull, 0, 0, 0]], dtype=np.float32))
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "adhesion 0.5 x weight", 0.5)
    simulate(rec, "adhesion 1.5 x weight", 1.5)
    simulate(rec, "1.5 x weight, then pulled down", 1.5, pull_at=0.6)
    rec.save()
