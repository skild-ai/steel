# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Replicated worlds with per-world reset of state and solver history."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 4
RESET_AT = [1.5, 2.5, 3.5]  # s; each reset re-drops world 0 or world 2


def build():
    scene = newton.ModelBuilder()
    for i in range(3):
        box = scene.add_body(
            xform=wp.transform((0.0, 0.0, 0.3 + 0.2 * i), wp.quat_rpy(0.3 * i, 0.4, 0.0))
        )
        scene.add_shape_box(box, hx=0.06, hy=0.05, hz=0.04)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(scene, WORLDS, spacing=(0.0, 0.45, 0.0))
    return builder.finalize()


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    bodies_per_world = model.body_count // WORLDS

    run = rec.variant(label, model)
    for frame in range(rec.frames(5.0)):
        t = frame / rec.fps
        if any(abs(t - r) < 0.5 / rec.fps for r in RESET_AT):
            world = 0 if t < 2.0 or t > 3.0 else 2
            # restore the world's start coordinates, then reset that world only
            q, qd = state_0.joint_q.numpy(), state_0.joint_qd.numpy()
            q_world = slice(world * bodies_per_world * 7, (world + 1) * bodies_per_world * 7)
            qd_world = slice(world * bodies_per_world * 6, (world + 1) * bodies_per_world * 6)
            q[q_world], qd[qd_world] = model.joint_q.numpy()[q_world], 0.0
            state_0.joint_q.assign(q)
            state_0.joint_qd.assign(qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            mask = np.zeros(WORLDS + 1, dtype=bool)  # the last entry selects global entities
            mask[world] = True
            solver.reset(state_0, world_mask=wp.array(mask, dtype=wp.bool))
        z = state_0.body_q.numpy()[:, 2].reshape(WORLDS, -1).max(axis=1)
        run.frame(state_0, **{f"world{w}_top": z[w] for w in range(WORLDS)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "four replicated worlds, masked resets")
    rec.save()
