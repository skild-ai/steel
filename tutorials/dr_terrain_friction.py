# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Terrain friction randomization: one floor shape per world, each with its own mu."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5
SPEED = 2.0  # m/s, initial slide speed


def build():
    world = newton.ModelBuilder()
    # a shared ground plane is one shape in world -1, so it has one mu for every world
    world.add_shape_box(
        -1, xform=wp.transform((0.8, 0.0, -0.05), wp.quat_identity()), hx=1.2, hy=0.12, hz=0.05
    )
    box = world.add_body(xform=wp.transform((-0.3, 0.0, 0.05), wp.quat_identity()))
    world.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=1.0))
    world.joint_qd[0] = SPEED  # free joint qd = (v_com, omega)
    builder = newton.ModelBuilder()
    builder.replicate(world, WORLDS, spacing=(0.0, 0.3, 0.0))
    return builder.finalize()


def simulate(rec, label, mu, combine):
    model = build()
    solver = SolverSteel(model, friction_combine=combine)
    floors = model.shape_body.numpy() == -1
    values = model.shape_material_mu.numpy()
    values[floors] = mu[model.shape_world.numpy()[floors]]
    model.shape_material_mu.assign(values)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    start = state_0.body_q.numpy()[:, 0].copy()

    run = rec.variant(label, model)
    for _ in range(rec.frames(3.0)):
        slide = state_0.body_q.numpy()[:, 0] - start
        run.frame(state_0, **{f"floor mu {m:.3f}": d for m, d in zip(mu, slide)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    mu = np.sort(rng.uniform(0.15, 0.8, WORLDS))
    rec = Recorder()
    # "min" lets the floor's mu decide against the box's 1.0
    simulate(rec, 'friction_combine="min"', mu, "min")
    # the default averages the pair: the effective mu is (floor + 1.0) / 2
    simulate(rec, 'friction_combine="average" (default)', mu, "average")
    rec.save()
    print("expected slide, m:", np.round(SPEED**2 / (2.0 * 9.81 * mu), 2))
