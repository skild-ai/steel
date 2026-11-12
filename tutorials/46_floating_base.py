# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""A floating-base articulation: a free root joint carries a body on four driven hips."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
STAND = 0.6  # rad, the knee target


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.9))
    torso = builder.add_link(xform=tf(0.0, 0.0, 0.5))
    builder.add_shape_box(
        torso, hx=0.25, hy=0.12, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=500.0)
    )
    joints = [builder.add_joint_free(torso)]
    leg_cfg = newton.ModelBuilder.ShapeConfig(density=800.0, mu=0.9)
    for sx in (-1, 1):
        for sy in (-1, 1):
            hip = (0.2 * sx, 0.14 * sy, 0.0)
            leg = builder.add_link(xform=tf(0.2 * sx, 0.14 * sy, 0.38))
            builder.add_shape_capsule(leg, radius=0.025, half_height=0.09, cfg=leg_cfg)
            joints.append(
                builder.add_joint_revolute(
                    torso,
                    leg,
                    axis=newton.Axis.Y,
                    parent_xform=tf(*hip),
                    child_xform=tf(0.0, 0.0, 0.12),
                    target_ke=150.0,
                    target_kd=5.0,
                    armature=0.01,
                )
            )
    builder.add_articulation(joints)
    return builder.finalize()


def simulate(rec, label, **options):
    model = build()
    # four feet with several contacts each need more than the default 32 articulated rows
    solver = SolverSteel(model, articulation_constraint_max=96, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for frame in range(rec.frames(5.0)):
        t = frame / rec.fps
        # splay the legs after landing, then sway them back and forth
        target = 0.0 if t < 1.0 else STAND * min(1.0, t - 1.0) * np.cos(2.0 * (t - 1.0))
        # targets are per joint coordinate: the free joint's 7 come first, then the 4 hips
        hips = np.array([target, target, -target, -target], dtype=np.float32)
        control.joint_target_q.assign(np.concatenate([np.zeros(7, dtype=np.float32), hips]))
        run.frame(state_0, torso_z=state_0.body_q.numpy()[0, 2])
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "free root joint + four driven hips")
    rec.save()
