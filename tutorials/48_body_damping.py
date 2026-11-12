"""Per-body linear and angular damping of free bodies."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
DAMPING = [
    (0.0, 0.0),
    (0.0, 0.05),
    (0.5, 0.5),
    (2.0, 2.0),
]  # (linear, angular) 1/s; (0, 0.05) is the default


def build():
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverSteel.register_custom_attributes(builder)
    for i in range(len(DAMPING)):
        box = builder.add_body(xform=wp.transform((-0.6, 0.3 * i - 0.45, 0.4), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.08, hy=0.04, hz=0.02)
    model = builder.finalize()
    model.steel.rigid_body_linear_damping.assign(
        np.array([d[0] for d in DAMPING], dtype=np.float32)
    )
    model.steel.rigid_body_angular_damping.assign(
        np.array([d[1] for d in DAMPING], dtype=np.float32)
    )
    model.joint_qd.assign(
        np.tile(np.array([0.6, 0.0, 0.0, 0.0, 0.0, 6.0], dtype=np.float32), len(DAMPING))
    )
    return model


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(3.0)):
        spin = state_0.body_qd.numpy()[:, 5]
        run.frame(state_0, **{f"lin{d[0]:g}_ang{d[1]:g}": spin[i] for i, d in enumerate(DAMPING)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "four damping settings, no gravity")
    rec.save()
