# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Contact sensors on Steel contacts, and what sleeping does to their readings."""

import newton
import warp as wp
from newton.sensors import SensorContact
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build():
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.7)
    builder.add_ground_plane(cfg=cfg)
    for k in range(3):
        for i in range(4):
            label = f"base{k}" if i == 0 else f"box{k}_{i}"
            box = builder.add_body(
                xform=wp.transform((0.3 * k - 0.3, 0.0, 0.05 + 0.1005 * i), wp.quat_identity()),
                label=label,
            )
            builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    return builder.finalize()


def simulate(rec, label, **options):
    model = build()
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    # one sensor per tower base, split by counterpart: column 0 is the ground plane (shape 0)
    sensor = SensorContact(
        model, sensing_bodies="base*", counterpart_shapes=[0], request_contact_attributes=False
    )
    kinds = set(sensor.solver_observable_kinds) | (
        {"body_awake"} if options.get("enable_sleeping") else set()
    )
    observed = solver.observables(kinds, contacts=contacts)
    state_0, state_1, control = model.state(), model.state(), model.control()
    weight = float(model.body_mass.numpy()[0]) * 9.81

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        # ground force on a base = the weight of its 4-box tower
        reading = sensor.force_matrix.numpy()[:, 0, 2].mean()
        awake = observed.body_awake if options.get("enable_sleeping") else None
        run.frame(state_0, awake=awake, base_force_n=reading, tower_weight_n=4 * weight)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT, observables=observed)
            state_0, state_1 = state_1, state_0
            sensor.update(state_0, contacts, observables=observed)
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "no sleeping")
    # sleeping islands skip their contact rows, so their contact forces read zero
    simulate(rec, "enable_sleeping=True", enable_sleeping=True)
    # keep the rows of sleeping islands: readings stay, the speed-up shrinks
    simulate(
        rec,
        "sleeping, sleep_skip_constraints=False",
        enable_sleeping=True,
        sleep_skip_constraints=False,
    )
    rec.save()
