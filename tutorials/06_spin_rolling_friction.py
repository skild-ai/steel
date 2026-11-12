"""Spin and rolling friction on free bodies with the per-shape steel:condim attribute."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build(condim):
    builder = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(
        density=800.0, mu=0.6, mu_torsional=0.01, mu_rolling=0.004
    )
    builder.add_ground_plane(cfg=cfg)
    spinner = builder.add_body(xform=wp.transform((0.0, 0.0, 0.08), wp.quat_identity()))
    builder.add_shape_sphere(spinner, radius=0.08, cfg=cfg)
    marker = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
    builder.add_shape_box(
        spinner,
        xform=wp.transform((0.0, 0.0, 0.08), wp.quat_identity()),
        hx=0.06,
        hy=0.008,
        hz=0.004,
        cfg=marker,
    )
    roller = builder.add_body(xform=wp.transform((-0.6, 0.35, 0.08), wp.quat_identity()))
    builder.add_shape_sphere(roller, radius=0.08, cfg=cfg)
    model = builder.finalize()
    model.steel.condim.fill_(condim)
    # free-joint velocity is (linear, angular): spin one ball, roll the other
    model.joint_qd.assign(np.array([0, 0, 0, 0, 0, 25.0, 1.0, 0, 0, 0, 0, 0], dtype=np.float32))
    return model


def simulate(rec, label, condim):
    model = build(condim)
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(5.0)):
        qd = state_0.body_qd.numpy()
        run.frame(state_0, spin=qd[0, 5], roll=np.linalg.norm(qd[1, :3]))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "condim 3 (sliding only, default)", 3)
    simulate(rec, "condim 4 (+ spin)", 4)
    simulate(rec, "condim 6 (+ spin + rolling)", 6)
    rec.save()
