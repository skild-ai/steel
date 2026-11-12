"""Speculative contacts: a contact gap that covers one step of travel stops a fast ball.

The shape gap decides which contacts the pipeline reports;
contact_gap_gate decides which of them Steel solves.
"""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 1  # record every step: the ball covers 17 cm per step


def build(gap):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(gap=gap)
    builder.add_shape_box(
        -1,
        xform=wp.transform((0.6, 0.0, 0.3), wp.quat_identity()),
        hx=0.005,
        hy=0.25,
        hz=0.25,
        cfg=cfg,
    )
    ball = builder.add_body(xform=wp.transform((-0.8, 0.0, 0.3), wp.quat_identity()))
    builder.add_shape_sphere(
        ball, radius=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=2000.0, gap=gap)
    )
    model = builder.finalize()
    model.joint_qd.assign(np.array([40.0, 0, 0, 0, 0, 0], dtype=np.float32))
    return model


def simulate(rec, label, gap, **options):
    model = build(gap)
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(0.25)):
        run.frame(state_0, x=state_0.body_q.numpy()[0, 0])
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder(fps=240)
    simulate(rec, "gap 5 mm: tunnels", 0.005)
    simulate(rec, "gap 20 cm (scale 1, default)", 0.2)
    simulate(rec, "gap 20 cm, contact_speculative_scale=0", 0.2, contact_speculative_scale=0.0)
    simulate(rec, "gap 20 cm, contact_gap_gate=0.01", 0.2, contact_gap_gate=0.01)
    rec.save()
