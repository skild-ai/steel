# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Conveyor belts: a static mesh with per-vertex surface velocity carries boxes through friction."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def belt_mesh(hx, hy, hz):
    """A closed box mesh with counter-clockwise outward winding."""
    v = np.array(
        [[x, y, z] for z in (-hz, hz) for y in (-hy, hy) for x in (-hx, hx)], dtype=np.float32
    )
    faces = [
        (0, 2, 1),
        (1, 2, 3),
        (4, 5, 6),
        (5, 7, 6),  # bottom, top
        (0, 1, 4),
        (1, 5, 4),
        (2, 6, 3),
        (3, 6, 7),  # -y, +y
        (0, 4, 2),
        (2, 4, 6),
        (1, 3, 5),
        (3, 7, 5),  # -x, +x
    ]
    return newton.Mesh(v, np.array(faces, dtype=np.int32).reshape(-1), enable_surface_velocity=True)


def build(speeds):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    belts = []
    for i, speed in enumerate(speeds):
        y = 0.3 * i - 0.3 * (len(speeds) - 1) / 2
        mesh = belt_mesh(0.6, 0.1, 0.02)
        builder.add_shape_mesh(
            -1,
            mesh=mesh,
            xform=wp.transform((0.0, y, 0.2), wp.quat_identity()),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8),
        )
        belts.append((mesh, speed))
        for k in range(3):
            box = builder.add_body(
                xform=wp.transform((-0.45 + 0.15 * k, y, 0.26), wp.quat_identity())
            )
            builder.add_shape_box(
                box,
                hx=0.04,
                hy=0.04,
                hz=0.04,
                cfg=newton.ModelBuilder.ShapeConfig(density=400.0, mu=0.8),
            )
    model = builder.finalize()
    for mesh, speed in belts:
        # every vertex moves along +x; Steel uses the part tangent to each contact
        mesh.mesh.velocities.assign(np.tile([speed, 0.0, 0.0], (8, 1)).astype(np.float32))
    return model


def simulate(rec, label, speeds):
    model = build(speeds)
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(3.0)):
        qd = state_0.body_qd.numpy()
        run.frame(state_0, **{f"belt_{s}": qd[3 * i + 1, 0] for i, s in enumerate(speeds)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "belts at 0, 0.1 and 0.25 m/s", [0.0, 0.1, 0.25])
    rec.save()
