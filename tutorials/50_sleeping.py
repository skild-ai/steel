# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Sleeping islands: settled towers fall asleep and skip their work until something wakes them."""

from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
KICK_AT = 7.0  # s
WORLDS = 64
CFG = newton.ModelBuilder.ShapeConfig(mu=0.7)


def add_towers(builder):
    for k in range(4):
        for i in range(5):
            box = builder.add_body(
                xform=wp.transform((0.5 * k - 0.75, 0.0, 0.05 + 0.1005 * i), wp.quat_identity())
            )
            builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=CFG)


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=CFG)
    add_towers(builder)
    ball = builder.add_body(xform=wp.transform((-0.25, -1.6, 0.08), wp.quat_identity()))
    builder.add_shape_sphere(
        ball, radius=0.08, cfg=newton.ModelBuilder.ShapeConfig(density=3000.0, mu=0.3)
    )
    return builder.finalize(), ball


def simulate(rec, label):
    model, ball = build()
    solver = SolverSteel(model, enable_sleeping=True)  # thresholds: 0.05 m/s, 0.15 rad/s for 0.5 s
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    observed = solver.observables({"body_awake"})
    state_0, state_1, control = model.state(), model.state(), model.control()
    kick = np.zeros((model.body_count, 6), dtype=np.float32)
    # one step of this force gives the ball 4 m/s toward the second tower
    kick[ball, 1] = model.body_mass.numpy()[ball] * 4.0 / DT

    run = rec.variant(label, model, scene="one")
    for frame in range(rec.frames(10.0)):
        awake = observed.body_awake
        run.frame(state_0, awake=awake, awake_bodies=awake.numpy().sum())
        for step in range(SUBSTEPS):
            state_0.clear_forces()
            if frame == int(KICK_AT * rec.fps) and step == 0:
                state_0.body_f.assign(kick)  # a nonzero force also wakes the ball's island
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT, observables=observed)
            state_0, state_1 = state_1, state_0
    run.check(solver)


def simulate_worlds(rec, deterministic):
    """WORLDS copies of the towers; the replay shows world 0, the chart counts every world."""
    towers = newton.ModelBuilder()
    add_towers(towers)
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=CFG)
    builder.replicate(towers, WORLDS)
    model = builder.finalize()
    shown = newton.ModelBuilder()
    shown.add_ground_plane(cfg=CFG)
    add_towers(shown)
    shown = shown.finalize()

    solver = SolverSteel(model, enable_sleeping=True)
    # the default pipeline can order contacts differently on each collide; this one sorts them
    pipeline = newton.CollisionPipeline(model, deterministic=deterministic)
    contacts = pipeline.contacts()
    observed = solver.observables({"body_awake"})
    state_0, state_1, control = model.state(), model.state(), model.control()

    pipe = "CollisionPipeline(deterministic=True)" if deterministic else "default pipeline"
    run = rec.variant(f"{WORLDS} worlds, {pipe}", shown, scene="worlds")
    for _ in range(rec.frames(6.0)):
        awake = observed.body_awake
        asleep_pct = 100.0 * (1.0 - awake.numpy().mean())
        first = SimpleNamespace(body_q=state_0.body_q[: shown.body_count])
        run.frame(first, awake=awake[: shown.body_count], asleep_pct=asleep_pct)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT, observables=observed)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "enable_sleeping=True")
    for deterministic in [False, True]:
        simulate_worlds(rec, deterministic)
    rec.save()
