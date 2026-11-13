# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Effort-limit randomization: per-world joint_effort_limit decides which arms lift."""

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
    cfg = newton.ModelBuilder.ShapeConfig(density=4000.0)
    arm.add_shape_sphere(link, xform=tf(0.3, 0.0, 0.0), radius=0.05, cfg=cfg)  # 2.1 kg payload
    j = arm.add_joint_revolute(
        -1,
        link,
        axis=(0.0, -1.0, 0.0),  # positive angles raise the arm
        parent_xform=tf(0.0, 0.0, 0.6),
        target_ke=500.0,
        target_kd=20.0,
    )
    arm.add_articulation([j])
    arm.joint_q[-1] = math.radians(-80.0)  # start hanging
    builder = newton.ModelBuilder()
    builder.replicate(arm, WORLDS, spacing=(0.0, 0.35, 0.0))
    return builder.finalize()


def simulate(rec, label, limit):
    model = build()
    solver = SolverSteel(model)
    model.joint_effort_limit.assign(limit.astype(np.float32))  # N m, one DOF per world
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control.joint_target_q.zero_()  # lift to horizontal

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, **{f"limit {e:.1f} N m": v for e, v in zip(limit, q)})
        for _ in range(SUBSTEPS):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    limit = np.sort(rng.uniform(2.0, 10.0, WORLDS))
    rec = Recorder()
    simulate(rec, "effort limit sampled per world", limit)
    rec.save()
    model = build()
    torque = 9.81 * float((model.body_mass.numpy() * model.body_com.numpy()[:, 0])[0])
    print(f"gravity torque at horizontal: {torque:.2f} N m")
