# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Joint velocity limits: a flywheel under constant torque."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build():
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    hub = builder.add_link(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_cylinder(
        hub, radius=0.2, half_height=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=500.0)
    )
    builder.add_shape_box(
        hub,
        xform=wp.transform((0.15, 0.0, 0.03), wp.quat_identity()),
        hx=0.04,
        hy=0.02,
        hz=0.01,
        cfg=newton.ModelBuilder.ShapeConfig(density=50.0),
    )
    j = builder.add_joint_revolute(
        -1,
        hub,
        axis=newton.Axis.Z,
        parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        velocity_limit=6.0,
    )
    builder.add_articulation([j])
    return builder.finalize()


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    control.joint_f.assign(np.array([3.0], dtype=np.float32))  # 3 N m, constant

    run = rec.variant(label, model)
    for _ in range(rec.frames(3.0)):
        run.frame(state_0, omega=state_0.joint_qd.numpy()[0])
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "velocity limits off (default)")
    simulate(rec, "enable_joint_velocity_limits=True", enable_joint_velocity_limits=True)
    rec.save()
