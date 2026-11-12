# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Free-body speed caps with steel:rigid_body_max_linear_velocity."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
MAX_SPEED = [float("inf"), 2.0, 1.0]  # m/s


def build():
    builder = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(builder)
    builder.add_ground_plane()
    for i in range(len(MAX_SPEED)):
        ball = builder.add_body(xform=wp.transform((0.0, 0.3 * i - 0.3, 1.5), wp.quat_identity()))
        builder.add_shape_sphere(ball, radius=0.06)
    model = builder.finalize()
    model.steel.rigid_body_max_linear_velocity.assign(np.array(MAX_SPEED, dtype=np.float32))
    return model


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(2.0)):
        speed = np.linalg.norm(state_0.body_qd.numpy()[:, :3], axis=1)
        run.frame(state_0, **{f"cap_{c:g}": speed[i] for i, c in enumerate(MAX_SPEED)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "unbounded | 2 m/s | 1 m/s")
    rec.save()
