# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-contact forces from the contact_f observable, and a unique indeterminate load split."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    table = builder.add_body(xform=wp.transform((0.0, 0.0, 0.2501), wp.quat_identity()))
    cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.6)
    builder.add_shape_box(
        table,
        xform=wp.transform((0.0, 0.0, 0.02), wp.quat_identity()),
        hx=0.3,
        hy=0.2,
        hz=0.02,
        cfg=cfg,
    )
    legs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            legs.append(
                builder.add_shape_box(
                    table,
                    xform=wp.transform((0.27 * sx, 0.17 * sy, -0.125), wp.quat_identity()),
                    hx=0.02,
                    hy=0.02,
                    hz=0.125,
                    cfg=cfg,
                )
            )
    load = builder.add_body(xform=wp.transform((0.15, 0.05, 0.36), wp.quat_identity()))
    builder.add_shape_box(
        load, hx=0.06, hy=0.06, hz=0.06, cfg=newton.ModelBuilder.ShapeConfig(density=3000.0)
    )
    return builder.finalize(), legs


def simulate(rec, label, **options):
    model, legs = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    observed = solver.observables({"contact_f"}, contacts=contacts)
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(3.0)):
        n = contacts.rigid_contact_count.numpy()[0]
        force = observed.contact_f.numpy()[:n]
        pairs = zip(
            contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]
        )
        leg_force = [0.0] * 4
        for k, (a, b) in enumerate(pairs):
            for shape in (a, b):
                if shape in legs:
                    leg_force[legs.index(shape)] += abs(force[k][2])
        run.frame(state_0, **{f"leg{i}": f for i, f in enumerate(leg_force)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT, observables=observed)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "contact_regularization=0 (default)")
    simulate(rec, "contact_regularization=0.02", contact_regularization=0.02)
    # converged: rigid PGS stops at an arbitrary split, regularization at the unique one
    simulate(rec, "regularization 0, iterations=48", iterations=48)
    simulate(rec, "regularization 0.2, iterations=48", contact_regularization=0.2, iterations=48)
    rec.save()
