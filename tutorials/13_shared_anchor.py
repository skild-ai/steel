"""Shared contact anchors: apply contact rows at the midpoint of the witness points."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 1  # one step per 240 Hz frame


def build(depth):
    """A box sliding at 2 m/s on a resting slab, starting ``depth`` deep in it."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    slab = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_box(
        slab, hx=0.3, hy=0.15, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=0.8)
    )
    box = builder.add_body(xform=wp.transform((-0.1, 0.0, 0.15 - depth), wp.quat_identity()))
    builder.add_shape_box(
        box, hx=0.08, hy=0.08, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=0.8)
    )
    model = builder.finalize()
    model.joint_qd.assign(np.array([0, 0, 0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0], dtype=np.float32))
    return model


def simulate(rec, label, depth, **options):
    model = build(depth)
    # point friction: patch anchors already act at their own anchor points
    solver = SolverSteel(model, friction_anchor_beta=0.0, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(0.6)):
        run.frame(state_0, box_pitch_rate=abs(state_0.body_qd.numpy()[1, 4]))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder(fps=240)
    simulate(rec, "touching, default", 0.0)
    simulate(rec, "3 cm deep, default", 0.03)
    simulate(
        rec, "3 cm deep, contact_friction_shared_anchor", 0.03, contact_friction_shared_anchor=True
    )
    simulate(rec, "3 cm deep, contact_shared_anchor", 0.03, contact_shared_anchor=True)
    rec.save()
