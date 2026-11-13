# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Friction randomization: write per-world shape_material_mu, then notify SHAPE_PROPERTIES."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5
SLOPE = math.radians(25.0)  # boxes slide where mu < tan(25 deg) = 0.47


def build():
    ramp = newton.ModelBuilder()
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), SLOPE)
    ramp.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.4), tilt), hx=0.9, hy=0.1, hz=0.03)
    downhill = np.array(wp.quat_rotate(tilt, wp.vec3(1.0, 0.0, 0.0)))
    normal = np.array(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, 1.0)))
    box = ramp.add_body(xform=wp.transform((0.0, 0.0, 0.4) - 0.7 * downhill + 0.081 * normal, tilt))
    ramp.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    builder = newton.ModelBuilder()
    builder.replicate(ramp, WORLDS, spacing=(0.0, 0.3, 0.0))
    builder.add_ground_plane()
    return builder.finalize()


def simulate(rec, label, mu, notify):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    start = state_0.body_q.numpy()[:, :3].copy()

    run = rec.variant(label, model)
    for frame in range(rec.frames(4.0)):
        if frame == rec.fps:  # randomize at 1 s, as an episode reset would
            # ramp and box of world w both get mu[w]; the ground plane (world -1) keeps its value
            world = model.shape_world.numpy()
            values = model.shape_material_mu.numpy()
            values[world >= 0] = mu[world[world >= 0]]
            model.shape_material_mu.assign(values)
            if notify:
                solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        slide = np.linalg.norm(state_0.body_q.numpy()[:, :3] - start, axis=1)
        run.frame(state_0, **{f"mu {m:.2f}": d for m, d in zip(mu, slide)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    mu = np.sort(rng.uniform(0.2, 0.8, WORLDS))
    rec = Recorder()
    simulate(rec, "mu sampled per world + notify", mu, True)
    simulate(rec, "mu sampled per world, no notify", mu, False)
    rec.save()
