# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Every rigid shape type Steel collides, falling into a triangle-mesh trough."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def trough_mesh(half_length=0.9, half_width=0.35, depth=0.18):
    """Two inclined quads meeting in a V along x, wound counter-clockwise seen from above."""
    v = np.array(
        [
            [-half_length, -half_width, depth],
            [half_length, -half_width, depth],
            [-half_length, 0.0, 0.0],
            [half_length, 0.0, 0.0],
            [-half_length, half_width, depth],
            [half_length, half_width, depth],
        ],
        dtype=np.float32,
    )
    faces = np.array([0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4], dtype=np.int32)
    return newton.Mesh(v, faces, is_solid=False)


def octahedron(r):
    v = np.array(
        [[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]], dtype=np.float32
    )
    faces = [0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5]
    return newton.Mesh(v, np.array(faces, dtype=np.int32))


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_shape_mesh(
        -1, mesh=trough_mesh(), xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity())
    )
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
    for i in range(7):
        q = wp.quat_rpy(0.4 * i, 0.7, 0.2 * i)
        body = builder.add_body(xform=wp.transform((0.25 * i - 0.75, 0.12 * (-1) ** i, 0.45), q))
        if i == 0:
            builder.add_shape_box(body, hx=0.06, hy=0.05, hz=0.04, cfg=cfg)
        elif i == 1:
            builder.add_shape_sphere(body, radius=0.06, cfg=cfg)
        elif i == 2:
            builder.add_shape_capsule(body, radius=0.04, half_height=0.07, cfg=cfg)
        elif i == 3:
            builder.add_shape_cylinder(body, radius=0.06, half_height=0.05, cfg=cfg)
        elif i == 4:
            builder.add_shape_cone(body, radius=0.07, half_height=0.07, cfg=cfg)
        elif i == 5:
            builder.add_shape_ellipsoid(body, rx=0.08, ry=0.05, rz=0.04, cfg=cfg)
        else:
            builder.add_shape_convex_hull(body, mesh=octahedron(0.08), cfg=cfg)
    return builder.finalize()


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        qd = state_0.body_qd.numpy()
        run.frame(
            state_0,
            max_speed=np.linalg.norm(qd[:, :3], axis=1).max(),
            lowest_z=state_0.body_q.numpy()[:, 2].min(),
        )
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "box, sphere, capsule, cylinder, cone, ellipsoid, convex hull")
    rec.save()
