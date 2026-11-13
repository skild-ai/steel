# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Passive joint springs (stiffness, rest_q) and viscous joint damping."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
JOINTS = [(0.0, 0.0), (6.0, 0.0), (6.0, 0.3), (6.0, 2.0)]  # (stiffness N m/rad, damping N m s/rad)


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for i, (stiffness, damping) in enumerate(JOINTS):
        y = 0.3 * i - 0.45  # each arm turns in its own xz plane
        link = builder.add_link(xform=tf(0.2, y, 0.4))
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
            axis=newton.Axis.Y,
            parent_xform=tf(0.0, y, 0.4),
            child_xform=tf(-0.2, 0.0, 0.0),
            stiffness=stiffness,
            damping=damping,
            rest_q=0.0,
        )
        builder.add_articulation([j])
    model = builder.finalize()
    model.joint_q.fill_(math.radians(60.0))
    model.joint_qd.fill_(-1.0)
    return model


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(5.0)):
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, **{f"k{k:g}_d{d:g}": q[i] for i, (k, d) in enumerate(JOINTS)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "free | spring | spring + light damping | spring + heavy damping")
    rec.save()
