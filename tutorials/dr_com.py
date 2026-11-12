# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Center-of-mass randomization: per-world body_com + BODY_INERTIAL_PROPERTIES."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5
HALF_WIDTH = 0.04  # the block tips once its COM leaves the footprint


def build():
    block = newton.ModelBuilder()
    body = block.add_body(xform=wp.transform((0.0, 0.0, 0.15), wp.quat_identity()))
    block.add_shape_box(body, hx=HALF_WIDTH, hy=HALF_WIDTH, hz=0.15)
    builder = newton.ModelBuilder()
    builder.replicate(block, WORLDS, spacing=(0.0, 0.3, 0.0))
    builder.add_ground_plane()
    return builder.finalize()


def simulate(rec, label, com_x, notify):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    state_0, state_1, control = model.state(), model.state(), model.control()
    run = rec.variant(label, model)
    for frame in range(rec.frames(4.0)):
        if frame == rec.fps:  # randomize at 1 s, as an episode reset would
            com = model.body_com.numpy()
            com[:, 0] = com_x[model.body_world.numpy()]  # body-frame offset; mass unchanged
            model.body_com.assign(com)
            if notify:
                solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        q = state_0.body_q.numpy()
        up = [wp.quat_rotate(wp.quat(*x[3:]), wp.vec3(0.0, 0.0, 1.0))[2] for x in q]
        tilt = np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))
        run.frame(state_0, **{f"com {c * 100:+.2f} cm": t for c, t in zip(com_x, tilt)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    com_x = np.sort(rng.uniform(-0.07, 0.07, WORLDS))
    rec = Recorder()
    simulate(rec, "COM shifted per world + notify", com_x, True)
    simulate(rec, "COM shifted per world, no notify", com_x, False)
    rec.save()
