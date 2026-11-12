"""Sleeping islands: settled towers fall asleep and skip their work until something wakes them."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
KICK_AT = 7.0  # s


def build():
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.7)
    builder.add_ground_plane(cfg=cfg)
    for k in range(4):
        for i in range(5):
            box = builder.add_body(
                xform=wp.transform((0.5 * k - 0.75, 0.0, 0.05 + 0.1005 * i), wp.quat_identity())
            )
            builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
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

    run = rec.variant(label, model)
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


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "enable_sleeping=True")
    rec.save()
