# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Hello Steel: drop a few shapes on the ground and step SolverSteel."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8  # 8 x 240 Hz steps per 30 Hz frame


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    shapes = ["box", "sphere", "capsule", "cylinder", "box", "sphere"]
    for i, kind in enumerate(shapes):
        body = builder.add_body(
            xform=wp.transform(
                (0.15 * (i % 2), 0.0, 0.3 + 0.25 * i), wp.quat_rpy(0.3 * i, 0.2, 0.1)
            )
        )
        if kind == "box":
            builder.add_shape_box(body, hx=0.08, hy=0.06, hz=0.05)
        elif kind == "sphere":
            builder.add_shape_sphere(body, radius=0.07)
        elif kind == "capsule":
            builder.add_shape_capsule(body, radius=0.04, half_height=0.1)
        else:
            builder.add_shape_cylinder(body, radius=0.06, half_height=0.06)
    return builder.finalize()


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        speed = np.linalg.norm(state_0.body_qd.numpy()[:, :3], axis=1).max()
        run.frame(state_0, max_speed=speed)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "SolverSteel(model), defaults")
    rec.save()
