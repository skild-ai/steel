# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Joint PD drives: position targets (target_ke, target_kd) and velocity targets (target_kd)."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
GAINS = [(50.0, 1.0), (50.0, 10.0), (400.0, 20.0)]  # (ke, kd) of the three position-driven arms


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def arm(builder, y, **drive):
    link = builder.add_link(xform=tf(0.2, y, 0.6))
    builder.add_shape_box(
        link, hx=0.2, hy=0.02, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=400.0)
    )
    builder.add_shape_sphere(
        link,
        xform=tf(0.2, 0.0, 0.0),
        radius=0.04,
        cfg=newton.ModelBuilder.ShapeConfig(density=2000.0),
    )
    j = builder.add_joint_revolute(
        -1,
        link,
        axis=(0.0, -1.0, 0.0),
        parent_xform=tf(0.0, y, 0.6),
        child_xform=tf(-0.2, 0.0, 0.0),
        **drive,
    )
    builder.add_articulation([j])


def build():
    builder = newton.ModelBuilder()
    for i, (ke, kd) in enumerate(GAINS):
        arm(builder, 0.3 * i - 0.45, target_ke=ke, target_kd=kd)
    arm(builder, 0.45, target_kd=5.0)  # velocity drive: no stiffness
    return builder.finalize()


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for frame in range(rec.frames(4.0)):
        t = frame / rec.fps
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(
            state_0,
            **{f"ke{ke:.0f}_kd{kd:.0f}": q[i] for i, (ke, kd) in enumerate(GAINS)},
            spin_rate=state_0.joint_qd.numpy()[3],
        )
        # step the position target to 45 degrees at 0.5 s; spin the last arm at 2 rad/s after 1 s
        control.joint_target_q.assign(
            np.array([math.radians(45.0) if t >= 0.5 else 0.0] * 3 + [0.0], dtype=np.float32)
        )
        control.joint_target_qd.assign(
            np.array([0.0, 0.0, 0.0, 2.0 if t >= 1.0 else 0.0], dtype=np.float32)
        )
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "three PD gains and one velocity drive")
    rec.save()
