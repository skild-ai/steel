# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""update_mass_matrix_interval: reuse the factored mass matrix for several steps to save work."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
LINKS = 6


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    builder = newton.ModelBuilder()
    parent, joints = -1, []
    for i in range(LINKS):
        link = builder.add_link(xform=tf(0.0, 0.0, 1.0 - 0.1 * i - 0.05))
        builder.add_shape_capsule(link, radius=0.02, half_height=0.03)
        anchor = tf(0.0, 0.0, 1.0) if parent < 0 else tf(0.0, 0.0, -0.05)
        # the armature tutorial's 0.001: at 240 Hz this light chain still gains energy
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=newton.Axis.Y,
                parent_xform=anchor,
                child_xform=tf(0.0, 0.0, 0.05),
                armature=0.001,
            )
        )
        builder.joint_q[-1] = 1.2 if i == 0 else 0.0
        parent = link
    builder.add_articulation(joints)
    return builder.finalize()


def swing_energy(model, state):
    """Energy above the chain hanging straight down at rest [J]."""
    rest = model.state()
    newton.eval_fk(model, wp.zeros_like(model.joint_q), model.joint_qd, rest)
    return energy(model, state) - energy(model, rest)


def energy(model, state):
    """Kinetic plus potential energy of the chain, including the joints' armature [J]."""
    mass, inertia = model.body_mass.numpy(), model.body_inertia.numpy()
    q, qd = state.body_q.numpy(), state.body_qd.numpy()
    total = 0.0
    for b in range(model.body_count):
        rot = np.array(wp.quat_to_matrix(wp.quat(*q[b, 3:]))).reshape(3, 3)
        omega = qd[b, 3:]
        total += (
            0.5 * mass[b] * qd[b, :3] @ qd[b, :3] + 0.5 * omega @ rot @ inertia[b] @ rot.T @ omega
        )
        total += mass[b] * 9.81 * q[b, 2]
    return total + 0.5 * (model.joint_armature.numpy() * state.joint_qd.numpy() ** 2).sum()


def simulate(rec, interval, substeps=SUBSTEPS):
    model = build()
    solver = SolverSteel(model, update_mass_matrix_interval=interval)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    start, swing = energy(model, state_0), swing_energy(model, state_0)
    dt = DT * SUBSTEPS / substeps

    label = f"update_mass_matrix_interval={interval}" + (" (default)" if interval == 1 else "")
    if substeps != SUBSTEPS:
        label += f", {round(1.0 / dt)} Hz steps"
    run = rec.variant(label, model)
    for _ in range(rec.frames(8.0)):
        energy_change = 100.0 * (energy(model, state_0) - start) / swing
        run.frame(state_0, tip_x=state_0.body_q.numpy()[-1, 0], energy_change_pct=energy_change)
        for _ in range(substeps):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    for interval in [1, 4, 16]:
        simulate(rec, interval)
    simulate(rec, 1, substeps=4 * SUBSTEPS)  # a quarter of the step, for reference
    rec.save()
