# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Inertia randomization: per-world body_inertia at fixed mass changes the natural frequency."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5
KE = 0.5  # torsional spring (a drive to 0): omega = sqrt(KE / I)


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    paddle = newton.ModelBuilder()
    link = paddle.add_link(xform=tf(0.0, 0.0, 0.5))
    paddle.add_shape_box(
        link, hx=0.12, hy=0.02, hz=0.02
    )  # hinged at its center, so no gravity torque
    paddle.add_shape_sphere(link, xform=tf(0.12, 0.0, 0.0), radius=0.04)
    paddle.add_shape_sphere(link, xform=tf(-0.12, 0.0, 0.0), radius=0.04)
    j = paddle.add_joint_revolute(
        -1, link, axis=newton.Axis.Z, parent_xform=tf(0.0, 0.0, 0.5), target_ke=KE
    )
    paddle.add_articulation([j])
    paddle.joint_q[-1] = 1.0
    builder = newton.ModelBuilder()
    builder.replicate(paddle, WORLDS, spacing=(0.0, 0.4, 0.0))
    return builder.finalize()


def simulate(rec, label, scale, notify, **options):
    model = build()
    solver = SolverSteel(model, **options)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for frame in range(rec.frames(5.0)):
        if frame == rec.fps:  # randomize at 1 s, as an episode reset would
            s = scale[model.body_world.numpy()][:, None, None]
            model.body_inertia.assign(model.body_inertia.numpy() * s)
            model.body_inv_inertia.assign(model.body_inv_inertia.numpy() / s)
            if notify:
                solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, **{f"inertia x{v:.2f}": a for v, a in zip(scale, q)})
        for _ in range(SUBSTEPS):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    scale = np.sort(rng.uniform(0.5, 3.0, WORLDS))
    rec = Recorder()
    simulate(rec, "inertia scaled per world + notify", scale, True)
    # the run is shorter than the interval, so without notify the mass matrix keeps the old inertia
    simulate(
        rec,
        "update_mass_matrix_interval=10000, no notify",
        scale,
        False,
        update_mass_matrix_interval=10000,
    )
    rec.save()
    i_z = build().body_inertia.numpy()[0, 2, 2]
    print("expected period, s:", np.round(2.0 * math.pi * np.sqrt(scale * i_z / KE), 2))
