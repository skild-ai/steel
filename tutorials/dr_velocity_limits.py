# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Velocity-limit randomization: per-world joint_velocity_limit, enabled at construction."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5
TORQUE = 0.05  # N m, constant on every rotor


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    rotor = newton.ModelBuilder()
    hub = rotor.add_link(xform=tf(0.0, 0.0, 0.5))
    rotor.add_shape_cylinder(
        hub, radius=0.12, half_height=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=500.0)
    )
    rotor.add_shape_box(hub, xform=tf(0.1, 0.0, 0.03), hx=0.03, hy=0.015, hz=0.01)  # a marker
    j = rotor.add_joint_revolute(-1, hub, axis=newton.Axis.Z, parent_xform=tf(0.0, 0.0, 0.5))
    rotor.add_articulation([j])
    builder = newton.ModelBuilder()
    builder.replicate(rotor, WORLDS, spacing=(0.0, 0.35, 0.0))
    return builder.finalize()


def simulate(rec, label, limit, enable):
    model = build()
    # the velocity-limit rows exist only if they are enabled here
    solver = SolverSteel(model, enable_joint_velocity_limits=enable)
    model.joint_velocity_limit.assign(limit.astype(np.float32))  # rad/s, one DOF per world
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
    state_0, state_1, control = model.state(), model.state(), model.control()
    control.joint_f.fill_(TORQUE)

    run = rec.variant(label, model)
    for _ in range(rec.frames(3.0)):
        omega = state_0.joint_qd.numpy()
        run.frame(state_0, **{f"limit {v:.1f} rad/s": w for v, w in zip(limit, omega)})
        for _ in range(SUBSTEPS):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(14)
    limit = np.sort(rng.uniform(2.0, 12.0, WORLDS))
    rec = Recorder()
    simulate(rec, "enable_joint_velocity_limits=True", limit, True)
    simulate(rec, "velocity limits off (default): ignored", limit, False)
    rec.save()
