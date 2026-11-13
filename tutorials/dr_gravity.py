# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Gravity randomization: per-world vectors with model.set_gravity, then notify MODEL_PROPERTIES."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5
LENGTH = 0.4


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    pendulum = newton.ModelBuilder()
    link = pendulum.add_link(xform=tf(0.0, 0.0, 0.8))
    pendulum.add_shape_capsule(
        link,
        xform=tf(0.0, 0.0, -0.2),
        radius=0.01,
        half_height=0.19,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0),
    )
    pendulum.add_shape_sphere(link, xform=tf(0.0, 0.0, -LENGTH), radius=0.04)
    j = pendulum.add_joint_revolute(-1, link, axis=newton.Axis.Y, parent_xform=tf(0.0, 0.0, 0.8))
    pendulum.add_articulation([j])
    pendulum.joint_q[-1] = math.radians(30.0)
    builder = newton.ModelBuilder()
    builder.replicate(pendulum, WORLDS, spacing=(0.0, 0.3, 0.0))
    return builder.finalize()


def simulate(rec, label, g):
    model = build()
    solver = SolverSteel(model)
    for w in range(WORLDS):
        model.set_gravity((0.0, 0.0, -g[w]), world=w)
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(5.0)):
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, **{f"g {v:.2f}": a for v, a in zip(g, q)})
        for _ in range(SUBSTEPS):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(10)
    g = np.sort(rng.uniform(2.0, 15.0, WORLDS))  # m/s^2
    rec = Recorder()
    simulate(rec, "gravity sampled per world", g)
    rec.save()
    print(
        "small-angle period of a point mass, s:", np.round(2.0 * math.pi * np.sqrt(LENGTH / g), 2)
    )
