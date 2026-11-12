# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Hydroelastic contacts, solved rigidly or with contact_compliance."""

import newton
import warp as wp
from newton.geometry import HydroelasticSDF
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build(hydroelastic, kh):
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(
        mu=0.6,
        is_hydroelastic=hydroelastic,
        kh=kh,
        gap=0.005,
        sdf_max_resolution=48,
        sdf_narrow_band_range=(-0.02, 0.02),
        sdf_padding=0.01,
    )
    builder.add_shape_box(
        -1,
        xform=wp.transform((0.0, 0.0, -0.03), wp.quat_identity()),
        hx=0.25,
        hy=0.25,
        hz=0.03,
        cfg=cfg,
    )
    heavy = cfg.copy()
    heavy.density = 4000.0
    sphere = builder.add_body(xform=wp.transform((-0.1, 0.0, 0.065), wp.quat_identity()))
    builder.add_shape_sphere(sphere, radius=0.06, cfg=heavy)
    box = builder.add_body(xform=wp.transform((0.1, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=heavy)
    # a third box dropped from 10 cm above the pad, to compare impact loads
    drop = builder.add_body(xform=wp.transform((0.0, 0.15, 0.15), wp.quat_identity()))
    builder.add_shape_box(drop, hx=0.05, hy=0.05, hz=0.05, cfg=heavy)
    model = builder.finalize()
    model.rigid_contact_max = 4096
    return model


def simulate(rec, label, hydroelastic, kh=1.0e7, **options):
    model = build(hydroelastic, kh)
    if hydroelastic:
        hydro = HydroelasticSDF.Config(
            reduce_contacts=True, moment_matching=True, anchor_contact=True, buffer_fraction=1.0
        )
        pipeline = newton.CollisionPipeline(
            model, rigid_contact_max=4096, sdf_hydroelastic_config=hydro
        )
        # a hydroelastic patch brings many contacts: raise the row capacities
        options.update(articulation_constraint_max=512, free_body_constraint_max=4096)
    else:
        pipeline = newton.CollisionPipeline(model)
    solver = SolverSteel(model, friction_anchor_beta=0.0, **options)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for _ in range(rec.frames(2.0)):
        q = state_0.body_q.numpy()
        sphere_mm, box_mm = 1000.0 * (q[0, 2] - 0.06), 1000.0 * (q[1, 2] - 0.05)
        impact_g = 0.0  # largest deceleration of the dropped box during the frame
        for _ in range(SUBSTEPS):
            vz = state_0.body_qd.numpy()[2, 2]
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            impact_g = max(impact_g, abs(state_0.body_qd.numpy()[2, 2] - vz) / DT / 9.81)
        run.frame(state_0, sphere_mm=sphere_mm, box_mm=box_mm, impact_g=impact_g)
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "point contacts, rigid", False)
    simulate(rec, "hydroelastic, rigid law", True)
    simulate(rec, "hydroelastic + contact_compliance, kh 1e7", True, 1.0e7, contact_compliance=True)
    simulate(rec, "hydroelastic + contact_compliance, kh 1e6", True, 1.0e6, contact_compliance=True)
    rec.save()
