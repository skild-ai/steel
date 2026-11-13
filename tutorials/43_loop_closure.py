# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Loop closures: a four-bar linkage as a chain plus one closing revolute joint."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
CRANK, COUPLER, ROCKER, GROUND = 0.12, 0.42, 0.30, 0.36


def joints_xz():
    """Pivot points A, B, C, D of the linkage with the crank at 60 degrees."""
    a, d = np.array([0.0, 0.0, 0.6]), np.array([GROUND, 0.0, 0.6])
    b = a + CRANK * np.array([math.cos(math.radians(60)), 0.0, math.sin(math.radians(60))])
    dist = np.linalg.norm(d - b)
    along = (COUPLER**2 - ROCKER**2 + dist**2) / (2 * dist)
    e = (d - b) / dist
    c = b + along * e + math.sqrt(COUPLER**2 - along**2) * np.array([-e[2], 0.0, e[0]])
    return a, b, c, d


def bar(p0, p1):
    """Center, orientation (x along the bar) and half length of a bar from p0 to p1."""
    v = p1 - p0
    return (
        tuple((p0 + p1) / 2),
        wp.quat_from_axis_angle(wp.vec3(0.0, -1.0, 0.0), math.atan2(v[2], v[0])),
        float(np.linalg.norm(v)) / 2,
    )


def build():
    builder = newton.ModelBuilder()
    a, b, c, d = joints_xz()
    cfg = newton.ModelBuilder.ShapeConfig(density=600.0)
    base = builder.add_link()
    builder.add_shape_box(
        base,
        xform=wp.transform((GROUND / 2, 0.0, 0.6), wp.quat_identity()),
        hx=GROUND / 2 + 0.03,
        hy=0.005,
        hz=0.01,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
    )
    links = []
    for p0, p1, width in ((a, b, 0.012), (b, c, 0.009), (d, c, 0.012)):
        center, rot, half = bar(p0, p1)
        link = builder.add_link(xform=wp.transform(center, rot))
        builder.add_shape_box(link, hx=half, hy=width, hz=width, cfg=cfg)
        links.append((link, rot, half))
    (crank, qa, ha), (coupler, qb, hb), (rocker, qc, hc) = links
    builder.add_shape_sphere(
        rocker,
        xform=wp.transform((hc, 0.0, 0.0), wp.quat_identity()),
        radius=0.04,
        cfg=newton.ModelBuilder.ShapeConfig(density=6000.0, has_shape_collision=False),
    )
    tree = [
        builder.add_joint_fixed(-1, base),
        builder.add_joint_revolute(
            base,
            crank,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(tuple(a), qa),
            child_xform=wp.transform((-ha, 0.0, 0.0), wp.quat_identity()),
            target_qd=15.0,
            target_kd=200.0,
        ),
        builder.add_joint_revolute(
            crank,
            coupler,
            axis=newton.Axis.Y,
            parent_xform=wp.transform((ha, 0.0, 0.0), wp.mul(wp.quat_inverse(qa), qb)),
            child_xform=wp.transform((-hb, 0.0, 0.0), wp.quat_identity()),
        ),
        builder.add_joint_revolute(
            coupler,
            rocker,
            axis=newton.Axis.Y,
            parent_xform=wp.transform((hb, 0.0, 0.0), wp.mul(wp.quat_inverse(qb), qc)),
            child_xform=wp.transform((hc, 0.0, 0.0), wp.quat_identity()),
        ),
    ]
    builder.add_articulation(tree)
    # the closing joint is not in the articulation: Steel solves it as five bilateral rows
    builder.add_joint_revolute(
        base,
        rocker,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(tuple(d), qc),
        child_xform=wp.transform((-hc, 0.0, 0.0), wp.quat_identity()),
    )
    return builder.finalize(), rocker, hc, d


def simulate(rec, label, **options):
    model, rocker, half, d = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        pose = state_0.body_q.numpy()[rocker]
        end = wp.transform_point(
            wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])), wp.vec3(-half, 0.0, 0.0)
        )
        run.frame(state_0, gap_mm=1000.0 * np.linalg.norm(np.array(end) - d))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "iterations=2", iterations=2)
    simulate(rec, "iterations=12 (default)")
    simulate(
        rec,
        "iterations=2, enable_bilateral_preelimination=True",
        iterations=2,
        enable_bilateral_preelimination=True,
    )
    rec.save()
