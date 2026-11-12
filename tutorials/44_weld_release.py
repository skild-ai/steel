# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""A FIXED loop closure (weld) released at runtime by writing joint_enabled."""

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
    builder.add_ground_plane()
    builder.add_shape_box(-1, xform=tf(-0.55, 0.0, 0.5), hx=0.05, hy=0.15, hz=0.5)
    builder.add_shape_box(-1, xform=tf(0.55, 0.0, 0.5), hx=0.05, hy=0.15, hz=0.5)
    plank = builder.add_link(xform=tf(0.0, 0.0, 0.98))
    builder.add_shape_box(
        plank, hx=0.5, hy=0.12, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=600.0)
    )
    hinge = builder.add_joint_revolute(
        -1,
        plank,
        axis=newton.Axis.Y,
        parent_xform=tf(-0.5, 0.0, 0.98),
        child_xform=tf(-0.5, 0.0, 0.0),
    )
    builder.add_articulation([hinge])
    # outside the articulation: a weld that closes the loop at the plank's right end
    weld = builder.add_joint_fixed(
        -1, plank, parent_xform=tf(0.5, 0.0, 0.98), child_xform=tf(0.5, 0.0, 0.0)
    )
    for i in range(3):
        box = builder.add_body(xform=tf(0.3 - 0.18 * i, 0.0, 1.06))
        builder.add_shape_box(
            box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=400.0)
        )
    return builder.finalize(), weld, plank


def simulate(rec, label):
    model, weld, plank = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for frame in range(rec.frames(3.5)):
        if frame == int(1.5 * rec.fps):
            enabled = model.joint_enabled.numpy()
            enabled[weld] = False  # read every step, also inside a captured CUDA graph
            model.joint_enabled.assign(enabled)
        run.frame(state_0, plank_z=state_0.body_q.numpy()[plank, 2])
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "weld released at 1.5 s")
    rec.save()
