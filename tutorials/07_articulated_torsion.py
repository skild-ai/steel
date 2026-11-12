# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Torsional friction on articulated contacts with steel:min_torsional_patch_radius."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build(radius):
    builder = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(builder)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    # a massless vertical slider carries a ball on a vertical hinge
    slider = builder.add_link(xform=wp.transform((0.0, 0.0, 0.06), wp.quat_identity()))
    j0 = builder.add_joint_prismatic(
        -1,
        slider,
        axis=newton.Axis.Z,
        parent_xform=wp.transform((0.0, 0.0, 0.06), wp.quat_identity()),
        armature=0.01,
    )
    ball = builder.add_link(xform=wp.transform((0.0, 0.0, 0.06), wp.quat_identity()))
    builder.add_shape_sphere(
        ball, radius=0.06, cfg=newton.ModelBuilder.ShapeConfig(density=3000.0, mu=0.6)
    )
    marker = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
    builder.add_shape_box(
        ball,
        xform=wp.transform((0.0, 0.0, 0.06), wp.quat_identity()),
        hx=0.05,
        hy=0.008,
        hz=0.004,
        cfg=marker,
    )
    j1 = builder.add_joint_revolute(slider, ball, axis=newton.Axis.Z)
    builder.add_articulation([j0, j1])
    model = builder.finalize()
    model.steel.min_torsional_patch_radius.fill_(radius)
    model.joint_qd.assign(np.array([0.0, 20.0], dtype=np.float32))
    return model


def simulate(rec, label, radius):
    model = build(radius)
    solver = SolverSteel(model, contact_torsion_device=True, friction_anchor_beta=0.0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        run.frame(state_0, spin=state_0.joint_qd.numpy()[1])
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "radius 0 (no torsion, default)", 0.0)
    simulate(rec, "radius 1 cm", 0.01)
    simulate(rec, "radius 5 cm", 0.05)
    rec.save()
