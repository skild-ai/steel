# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Armature randomization: per-world joint_armature + JOINT_DOF_INERTIAL_PROPERTIES."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    arm = newton.ModelBuilder()
    link = arm.add_link(xform=tf(0.0, 0.0, 0.6))
    arm.add_shape_box(link, xform=tf(0.15, 0.0, 0.0), hx=0.15, hy=0.02, hz=0.02)
    j = arm.add_joint_revolute(
        -1, link, axis=newton.Axis.Z, parent_xform=tf(0.0, 0.0, 0.6), target_ke=5.0, target_kd=0.2
    )
    arm.add_articulation([j])
    builder = newton.ModelBuilder()
    builder.replicate(arm, WORLDS, spacing=(0.0, 0.35, 0.0))
    return builder.finalize()


def simulate(rec, label, armature, notify):
    model = build()
    solver = SolverSteel(model)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for frame in range(rec.frames(4.0)):
        if frame == rec.fps:  # randomize at 1 s, then step the target to 90 degrees
            model.joint_armature.assign(armature.astype(np.float32))  # kg m^2, one DOF per world
            if notify:
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES)
            control.joint_target_q.fill_(math.radians(90.0))
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, **{f"armature {a:.3f}": v for a, v in zip(armature, q)})
        for _ in range(SUBSTEPS):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    armature = np.sort(rng.uniform(0.0, 0.05, WORLDS))
    rec = Recorder()
    simulate(rec, "armature sampled per world + notify", armature, True)
    simulate(rec, "armature sampled per world, no notify", armature, False)
    rec.save()
