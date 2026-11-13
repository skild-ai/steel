# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Solve modes: the serial sweep vs the colored sweep with the propagation contact response."""

import time

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build():
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
    builder.add_ground_plane(cfg=cfg)
    # a 3-link arm turning clockwise into a 5 x 5 grid of boxes
    base = builder.add_link(xform=wp.transform((0.0, -0.45, 0.06), wp.quat_identity()))
    builder.add_shape_box(base, hx=0.04, hy=0.04, hz=0.04, cfg=cfg)
    joints = [
        builder.add_joint_revolute(
            -1,
            base,
            axis=newton.Axis.Z,
            parent_xform=wp.transform((0.0, -0.45, 0.06), wp.quat_identity()),
            target_kd=50.0,
            target_qd=-0.8,
        )
    ]
    parent = base
    for i in range(3):
        link = builder.add_link(
            xform=wp.transform((0.0, -0.45 + 0.12 + 0.24 * i, 0.06), wp.quat_identity())
        )
        builder.add_shape_box(link, hx=0.02, hy=0.11, hz=0.03, cfg=cfg)
        anchor = (0.0, 0.0, 0.0) if i == 0 else (0.0, 0.12, 0.0)
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(anchor, wp.quat_identity()),
                child_xform=wp.transform((0.0, -0.12, 0.0), wp.quat_identity()),
                target_ke=60.0,
                target_kd=2.0,
            )
        )
        parent = link
    builder.add_articulation(joints)
    for i in range(5):
        for j in range(5):
            box = builder.add_body(
                xform=wp.transform((0.13 * i + 0.1, 0.13 * j - 0.36, 0.04), wp.quat_identity())
            )
            builder.add_shape_box(
                box,
                hx=0.04,
                hy=0.04,
                hz=0.04,
                cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.6),
            )
    return builder.finalize()


def build_stack(n=8, h=0.05):
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    for i in range(n):
        box = builder.add_body(
            xform=wp.transform((0.0, 0.0, h + 2 * h * i + 0.0002 * i), wp.quat_identity())
        )
        builder.add_shape_box(box, hx=h, hy=h, hz=h, cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    return builder.finalize()


def simulate_stack(rec, label, **options):
    """The same sag check as the iterations tutorial, to compare convergence of the two sweeps."""
    model = build_stack()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model, scene="stack")
    for _ in range(rec.frames(4.0)):
        run.frame(state_0, sag_mm=1000.0 * (0.75 - state_0.body_q.numpy()[-1, 2]))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(
        model, articulation_constraint_max=384, free_body_constraint_max=2048, **options
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model, scene="arm")
    elapsed = 0.0
    for frame in range(rec.frames(5.0)):
        moved = np.linalg.norm(
            state_0.body_q.numpy()[4:, :2] - model.body_q.numpy()[4:, :2], axis=1
        )
        run.frame(state_0, mean_push_mm=1000.0 * moved.mean())
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        if frame > 0:  # the first frame includes kernel compilation
            elapsed += time.perf_counter() - start
    run.check(solver)
    print(f"{label}: {1000.0 * elapsed / ((rec.frames(5.0) - 1) * SUBSTEPS):.3f} ms per eager step")


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, 'solve_mode="serial" (default)')
    simulate(
        rec,
        'solve_mode="colored" + "propagation"',
        solve_mode="colored",
        articulated_contact_response="propagation",
    )
    simulate_stack(rec, 'stack, "serial"')
    simulate_stack(
        rec, 'stack, "colored"', solve_mode="colored", articulated_contact_response="propagation"
    )
    rec.save()
