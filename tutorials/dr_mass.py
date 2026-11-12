# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Mass randomization: per-world body mass and inertia + BODY_INERTIAL_PROPERTIES."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5
KE, KD = 15.0, 1.0  # hinge drive holding the arm level


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    arm = newton.ModelBuilder()
    link = arm.add_link(xform=tf(0.0, 0.0, 0.8))
    arm.add_shape_box(link, xform=tf(0.2, 0.0, 0.0), hx=0.2, hy=0.02, hz=0.02)
    arm.add_shape_sphere(link, xform=tf(0.4, 0.0, 0.0), radius=0.05)
    j = arm.add_joint_revolute(
        -1, link, axis=newton.Axis.Y, parent_xform=tf(0.0, 0.0, 0.8), target_ke=KE, target_kd=KD
    )
    arm.add_articulation([j])
    builder = newton.ModelBuilder()
    builder.replicate(arm, WORLDS, spacing=(0.0, 0.35, 0.0))
    return builder.finalize()


def randomize_mass(model, scale):
    """Scale every body of world w by scale[w]; keep the inverse arrays consistent."""
    s = scale[model.body_world.numpy()]
    model.body_mass.assign(model.body_mass.numpy() * s)
    model.body_inv_mass.assign(model.body_inv_mass.numpy() / s)
    model.body_inertia.assign(model.body_inertia.numpy() * s[:, None, None])
    model.body_inv_inertia.assign(model.body_inv_inertia.numpy() / s[:, None, None])


def simulate(rec, label, scale):
    model = build()
    solver = SolverSteel(model)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for frame in range(rec.frames(4.0)):
        if frame == rec.fps:  # randomize at 1 s, as an episode reset would
            randomize_mass(model, scale)
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        sag = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, **{f"mass x{s:.2f}": a for s, a in zip(scale, sag)})
        for _ in range(SUBSTEPS):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    scale = np.sort(rng.uniform(0.5, 2.0, WORLDS))
    rec = Recorder()
    simulate(rec, "5 worlds, mass x0.5 to x2", scale)
    rec.save()
