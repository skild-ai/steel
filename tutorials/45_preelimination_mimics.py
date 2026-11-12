"""bilateral_preelimination_include_mimics: pre-eliminate closures with or without mimic rows."""

import math
import warnings

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
CRANK, COUPLER, ROCKER, GROUND = 0.12, 0.42, 0.30, 0.36
FOLLOWERS = (
    4  # with the five closure rows: nine bilateral rows, one more than pre-elimination holds
)


def joints_xz():
    """Pivot points A, B, C, D of the four-bar with the crank at 60 degrees."""
    a, d = np.array([0.0, 0.0, 0.6]), np.array([GROUND, 0.0, 0.6])
    b = a + CRANK * np.array([math.cos(math.radians(60)), 0.0, math.sin(math.radians(60))])
    dist = np.linalg.norm(d - b)
    along = (COUPLER**2 - ROCKER**2 + dist**2) / (2 * dist)
    e = (d - b) / dist
    c = b + along * e + math.sqrt(COUPLER**2 - along**2) * np.array([-e[2], 0.0, e[0]])
    return a, b, c, d


def bar(p0, p1):
    v = p1 - p0
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, -1.0, 0.0), math.atan2(v[2], v[0]))
    return tuple((p0 + p1) / 2), rot, float(np.linalg.norm(v)) / 2


def tf(x, y, z, q=None):
    return wp.transform((x, y, z), q if q is not None else wp.quat_identity())


def build():
    """A four-bar driven by its crank, plus four indicator arms that mimic the crank angle."""
    builder = newton.ModelBuilder()
    a, b, c, d = joints_xz()
    hidden = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
    base = builder.add_link()
    builder.add_shape_box(
        base,
        xform=tf(GROUND / 2 - 0.2, 0.0, 0.6),
        hx=GROUND / 2 + 0.23,
        hy=0.005,
        hz=0.01,
        cfg=hidden,
    )
    links = []
    for p0, p1, width in ((a, b, 0.012), (b, c, 0.009), (d, c, 0.012)):
        center, rot, half = bar(p0, p1)
        link = builder.add_link(xform=wp.transform(center, rot))
        builder.add_shape_box(
            link, hx=half, hy=width, hz=width, cfg=newton.ModelBuilder.ShapeConfig(density=600.0)
        )
        links.append((link, rot, half))
    (crank, qa, ha), (coupler, qb, hb), (rocker, qc, hc) = links
    heavy = newton.ModelBuilder.ShapeConfig(density=6000.0, has_shape_collision=False)
    builder.add_shape_sphere(rocker, xform=tf(hc, 0.0, 0.0), radius=0.04, cfg=heavy)
    root = builder.add_joint_fixed(-1, base)
    drive = builder.add_joint_revolute(
        base,
        crank,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(tuple(a), qa),
        child_xform=tf(-ha, 0.0, 0.0),
        target_qd=15.0,
        target_kd=200.0,
    )
    tree = [
        root,
        drive,
        builder.add_joint_revolute(
            crank,
            coupler,
            axis=newton.Axis.Y,
            parent_xform=wp.transform((ha, 0.0, 0.0), wp.mul(wp.quat_inverse(qa), qb)),
            child_xform=tf(-hb, 0.0, 0.0),
        ),
        builder.add_joint_revolute(
            coupler,
            rocker,
            axis=newton.Axis.Y,
            parent_xform=wp.transform((hb, 0.0, 0.0), wp.mul(wp.quat_inverse(qb), qc)),
            child_xform=tf(hc, 0.0, 0.0),
        ),
    ]
    for i in range(FOLLOWERS):
        x = -0.12 - 0.08 * i
        arm = builder.add_link(xform=tf(x, 0.0, 0.6))
        builder.add_shape_box(
            arm,
            xform=tf(0.03, 0.0, 0.0),
            hx=0.03,
            hy=0.006,
            hz=0.006,
            cfg=newton.ModelBuilder.ShapeConfig(density=2000.0, has_shape_collision=False),
        )
        tree.append(
            builder.add_joint_revolute(base, arm, axis=newton.Axis.Y, parent_xform=tf(x, 0.0, 0.6))
        )
    builder.add_articulation(tree)
    for joint in tree[4:]:
        builder.set_joint_mimic(joint, drive, coeffs=(0.0, 1.0))
    # the closing joint is not in the articulation: five bilateral rows
    builder.add_joint_revolute(
        base,
        rocker,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(tuple(d), qc),
        child_xform=tf(-hc, 0.0, 0.0),
    )
    return builder.finalize(), rocker, hc, d


def simulate(rec, label, **options):
    model, rocker, half, d = build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverSteel(model, iterations=2, **options)
    for warning in caught:
        print(f"{label}: {warning.message}")
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(3.0)):
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
    simulate(rec, "no pre-elimination")
    simulate(
        rec, "pre-elimination, include_mimics=True (default)", enable_bilateral_preelimination=True
    )
    simulate(
        rec,
        "pre-elimination, include_mimics=False",
        enable_bilateral_preelimination=True,
        bilateral_preelimination_include_mimics=False,
    )
    rec.save()
