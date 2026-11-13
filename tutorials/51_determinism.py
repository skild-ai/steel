# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Run-to-run determinism: the same pile poured twice, with and without deterministic row order."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build():
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    builder.add_ground_plane(cfg=cfg)
    for s in (-1, 1):
        builder.add_shape_box(
            -1,
            xform=wp.transform((0.22 * s, 0.0, 0.1), wp.quat_identity()),
            hx=0.01,
            hy=0.22,
            hz=0.1,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform((0.0, 0.22 * s, 0.1), wp.quat_identity()),
            hx=0.22,
            hy=0.01,
            hz=0.1,
        )
    rng = np.random.default_rng(3)
    for i in range(40):
        axis = rng.normal(size=3)
        q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), float(rng.uniform(0.0, np.pi)))
        body = builder.add_body(
            xform=wp.transform(
                (rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), 0.15 + 0.06 * i), q
            )
        )
        if i % 2:
            builder.add_shape_box(body, hx=0.03, hy=0.025, hz=0.02, cfg=cfg)
        else:
            builder.add_shape_sphere(body, radius=0.025, cfg=cfg)
    return builder.finalize()


def simulate(rec, deterministic, slot):
    """Pour twice and record both runs, the second as a translucent ghost over the first."""
    name = "deterministic=RUN_TO_RUN" if deterministic else "default"
    runs = []
    for k in range(2):
        model = build()
        mode = (
            wp.DeterministicMode.RUN_TO_RUN
            if deterministic
            else wp.DeterministicMode.NOT_GUARANTEED
        )
        # a deep pile needs more free-body rows than the default 512
        solver = SolverSteel(model, deterministic=mode, free_body_constraint_max=4096)
        # the contact order must be fixed too
        pipeline = newton.CollisionPipeline(model, deterministic=deterministic)
        contacts = pipeline.contacts()
        state_0, state_1, control = model.state(), model.state(), model.control()
        run = rec.variant(f"{name}, run {k + 1}", model, ghost=k == 1, slot=slot)
        for _ in range(rec.frames(4.0)):
            run.frame(state_0)
            for _ in range(SUBSTEPS):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, DT)
                state_0, state_1 = state_1, state_0
        run.check(solver)
        runs.append(run)
    # metric: the largest position difference between the two runs at each frame
    a, b = (np.stack(r.poses)[:, :, :3] for r in runs)
    divergence = 1000.0 * np.linalg.norm(a - b, axis=2).max(axis=1)
    for r in runs:
        r.metrics = [{"divergence_mm": round(float(d), 4)} for d in divergence]


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, False, 0)
    simulate(rec, True, 1)
    rec.save()
