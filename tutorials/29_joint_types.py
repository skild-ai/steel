# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Joint types: revolute, prismatic, ball, D6 and fixed joints in one articulation each."""

import math

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
Dof = newton.ModelBuilder.JointDofConfig


def tf(x, y, z, q=None):
    return wp.transform((x, y, z), q if q is not None else wp.quat_identity())


def pendulum_link(builder, x):
    link = builder.add_link(xform=tf(x, 0.0, 0.6))
    builder.add_shape_capsule(link, xform=tf(0.0, 0.0, -0.15), radius=0.02, half_height=0.13)
    builder.add_shape_box(link, xform=tf(0.0, 0.0, -0.3), hx=0.05, hy=0.05, hz=0.05)
    return link


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    anchor = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
    joints = []
    for i, kind in enumerate(["revolute", "prismatic", "ball", "d6", "fixed"]):
        x = 0.6 * i - 1.2
        builder.add_shape_sphere(-1, xform=tf(x, 0.0, 0.6), radius=0.025, cfg=anchor)
        link = pendulum_link(builder, x)
        pivot = tf(x, 0.0, 0.6)
        if kind == "revolute":
            j = builder.add_joint_revolute(-1, link, axis=newton.Axis.Y, parent_xform=pivot)
        elif kind == "prismatic":
            # slides along an axis tilted 0.5 rad from vertical
            j = builder.add_joint_prismatic(
                -1, link, axis=(math.sin(0.5), 0.0, -math.cos(0.5)), parent_xform=pivot
            )
        elif kind == "ball":
            j = builder.add_joint_ball(-1, link, parent_xform=pivot)
        elif kind == "d6":
            # one linear DOF along z and two rotations: a spring-hung pendulum
            j = builder.add_joint_d6(
                -1,
                link,
                parent_xform=pivot,
                linear_axes=[Dof(axis=newton.Axis.Z, stiffness=200.0, damping=2.0)],
                angular_axes=[Dof(axis=newton.Axis.X), Dof(axis=newton.Axis.Y)],
            )
        else:
            j = builder.add_joint_fixed(-1, link, parent_xform=pivot)
        builder.add_articulation([j])
        joints.append(j)
    model = builder.finalize()
    # start every joint away from rest so each one moves
    q = model.joint_q.numpy()
    starts = model.joint_q_start.numpy()
    q[starts[0]] = 1.0
    ball_q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 0.0)), 0.9)
    q[starts[2] : starts[2] + 4] = [ball_q[0], ball_q[1], ball_q[2], ball_q[3]]
    q[starts[3] + 1 : starts[3] + 3] = [0.6, 0.3]
    model.joint_q.assign(q)
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
        q = state_0.body_q.numpy()
        # height of each link's tip, 0.3 m down its local z axis
        tip_z = q[:, 2] - 0.3 * (1.0 - 2.0 * (q[:, 3] ** 2 + q[:, 4] ** 2))
        run.frame(
            state_0,
            **{
                name: tip_z[i]
                for i, name in enumerate(["revolute", "prismatic", "ball", "d6", "fixed"])
            },
        )
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "revolute | prismatic | ball | D6 | fixed")
    rec.save()
