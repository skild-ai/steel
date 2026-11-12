# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Joint armature: rotor inertia added to a DOF slows its response, also on fast light links."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
ARMATURE = [0.0, 0.02, 0.1]  # kg m^2
CHAIN_ARMATURE = [0.0, 0.001, 0.01]  # kg m^2


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    builder = newton.ModelBuilder()
    for i, armature in enumerate(ARMATURE):
        y = 0.45 * i - 0.45
        link = builder.add_link(xform=tf(0.0, y, 0.8))
        builder.add_shape_capsule(
            link,
            xform=tf(0.0, 0.0, -0.2),
            radius=0.02,
            half_height=0.18,
            cfg=newton.ModelBuilder.ShapeConfig(density=400.0),
        )
        builder.add_shape_sphere(
            link,
            xform=tf(0.0, 0.0, -0.4),
            radius=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(density=2000.0),
        )
        j = builder.add_joint_revolute(
            -1, link, axis=newton.Axis.Y, parent_xform=tf(0.0, y, 0.8), armature=armature
        )
        builder.add_articulation([j])
    model = builder.finalize()
    model.joint_q.fill_(math.radians(60.0))
    return model


def build_chain(armature):
    """Six light capsule links hanging from a hinge, released at 1.2 rad."""
    builder = newton.ModelBuilder()
    parent, joints = -1, []
    for i in range(6):
        link = builder.add_link(xform=tf(0.0, 0.0, 1.0 - 0.1 * i - 0.05))
        builder.add_shape_capsule(link, radius=0.02, half_height=0.03)
        anchor = tf(0.0, 0.0, 1.0) if parent < 0 else tf(0.0, 0.0, -0.05)
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=newton.Axis.Y,
                parent_xform=anchor,
                child_xform=tf(0.0, 0.0, 0.05),
                armature=armature,
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
    """Kinetic plus potential energy, including the joints' armature [J]."""
    mass, inertia = model.body_mass.numpy(), model.body_inertia.numpy()
    q, qd = state.body_q.numpy(), state.body_qd.numpy()
    total = 0.5 * (model.joint_armature.numpy() * state.joint_qd.numpy() ** 2).sum()
    for b in range(model.body_count):
        rot = np.array(wp.quat_to_matrix(wp.quat(*q[b, 3:]))).reshape(3, 3)
        omega = qd[b, 3:]
        total += (
            0.5 * mass[b] * qd[b, :3] @ qd[b, :3] + 0.5 * omega @ rot @ inertia[b] @ rot.T @ omega
        )
        total += mass[b] * 9.81 * q[b, 2]
    return total


def simulate_chain(rec, armature):
    model = build_chain(armature)
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    start, swing = energy(model, state_0), swing_energy(model, state_0)

    run = rec.variant(f"light chain, armature {armature:g}", model, scene="chain")
    for _ in range(rec.frames(8.0)):
        run.frame(state_0, energy_change_pct=100.0 * (energy(model, state_0) - start) / swing)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model, scene="pendulums")
    for _ in range(rec.frames(5.0)):
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, **{f"armature_{a:g}": q[i] for i, a in enumerate(ARMATURE)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "armature 0 | 0.02 | 0.1 kg m^2")
    for armature in CHAIN_ARMATURE:
        simulate_chain(rec, armature)
    rec.save()
