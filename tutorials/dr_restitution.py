"""Restitution randomization: per-world shape_material_restitution + SHAPE_PROPERTIES."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
WORLDS = 5
RADIUS = 0.06


def build():
    drop = newton.ModelBuilder()
    ball = drop.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    drop.add_shape_sphere(ball, radius=RADIUS, cfg=newton.ModelBuilder.ShapeConfig(restitution=0.0))
    builder = newton.ModelBuilder()
    builder.replicate(drop, WORLDS, spacing=(0.0, 0.25, 0.0))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(restitution=0.0))
    return builder.finalize()


def simulate(rec, label, e, notify):
    model = build()
    # "max" lets the ball's coefficient decide against the shared ground's 0
    solver = SolverSteel(model, restitution_combine="max")
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    world = model.shape_world.numpy()
    values = model.shape_material_restitution.numpy()
    values[world >= 0] = e[world[world >= 0]]
    model.shape_material_restitution.assign(values)
    if notify:
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        height = state_0.body_q.numpy()[:, 2] - RADIUS
        run.frame(state_0, **{f"e {v:.2f}": h for v, h in zip(e, height)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    e = np.sort(rng.uniform(0.1, 0.9, WORLDS))
    rec = Recorder()
    simulate(rec, "restitution sampled per world + notify", e, True)
    simulate(rec, "restitution sampled per world, no notify", e, False)
    rec.save()
