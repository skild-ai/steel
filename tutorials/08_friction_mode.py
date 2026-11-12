# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""friction_mode: the per-contact Coulomb update, checked against analytic results."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
MU, SPEED, G = 0.5, 2.0, 9.81


def build():
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=MU)
    builder.add_ground_plane(cfg=cfg)
    # a box sliding along x and a ball spinning without rolling (it slides until it rolls)
    box = builder.add_body(xform=wp.transform((-0.6, -0.15, 0.05), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    ball = builder.add_body(xform=wp.transform((-0.6, 0.15, 0.05), wp.quat_identity()))
    builder.add_shape_sphere(ball, radius=0.05, cfg=cfg)
    model = builder.finalize()
    model.joint_qd.assign(np.array([SPEED, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60.0, 0], dtype=np.float32))
    return model


def build_slope(ratio=0.97):
    """A box near the top of a 3 m slope, at ``ratio`` of its friction limit, above a floor."""
    builder = newton.ModelBuilder()
    angle = math.atan(ratio * MU)
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -angle)
    cfg = newton.ModelBuilder.ShapeConfig(mu=MU)
    builder.add_ground_plane(cfg=cfg)
    center = np.array([0.0, 0.0, 1.5 * math.sin(angle)])
    builder.add_shape_box(-1, xform=wp.transform(center, tilt), hx=1.5, hy=0.2, hz=0.05, cfg=cfg)
    normal = np.array(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, 1.0)))
    uphill = np.array(wp.quat_rotate(tilt, wp.vec3(1.0, 0.0, 0.0)))
    start = center + 1.2 * uphill + 0.11 * normal
    box = builder.add_body(xform=wp.transform(start, tilt))
    builder.add_shape_box(box, hx=0.12, hy=0.08, hz=0.06, cfg=cfg)
    return builder.finalize()


def simulate_slope(rec, mode):
    model = build_slope()
    solver = SolverSteel(model, friction_mode=mode, friction_anchor_beta=0.0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    start = state_0.body_q.numpy()[0, :3].copy()

    run = rec.variant(f'slope at 97% of the limit, "{mode}"', model, scene="slope")
    for _ in range(rec.frames(5.0)):
        run.frame(
            state_0, slope_slip_mm=1000.0 * np.linalg.norm(state_0.body_q.numpy()[0, :3] - start)
        )
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


def simulate(rec, mode):
    model = build()
    solver = SolverSteel(model, friction_mode=mode, friction_anchor_beta=0.0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    start = state_0.body_q.numpy()[0, 0]

    run = rec.variant(f'friction_mode="{mode}"', model, scene="slide")
    for _ in range(rec.frames(1.5)):
        q, qd = state_0.body_q.numpy(), state_0.body_qd.numpy()
        run.frame(state_0, slide_mm=1000.0 * (q[0, 0] - start), ball_v=qd[1, 0])
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    print(f"analytic stopping distance v^2 / (2 mu g) = {1000 * SPEED**2 / (2 * MU * G):.1f} mm")
    rec = Recorder()
    for mode in ["fixed_normal", "bisection", "coulomb_newton"]:
        simulate(rec, mode)
    for mode in ["fixed_normal", "bisection", "coulomb_newton"]:
        simulate_slope(rec, mode)
    rec.save()
