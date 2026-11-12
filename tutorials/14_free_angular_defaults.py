"""Model-wide spin and rolling friction for free bodies with one SolverSteel argument."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def build(condim=3):
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
    # spin one ball about the vertical, roll the other at 1 m/s
    model.joint_qd.assign(np.array([0, 0, 0, 0, 0, 25.0, 1.0, 0, 0, 0, 0, 0], dtype=np.float32))
    return model


def simulate(rec, label, condim=3, **options):
    model = build(condim)
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        qd = state_0.body_qd.numpy()
        run.frame(state_0, spin=qd[0, 5], roll=np.linalg.norm(qd[1, :3]))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "condim 3, no override (default)")
    simulate(rec, "per-shape steel:condim 6", condim=6)
    # one argument per coefficient, for every free-body contact, without touching shape data
    simulate(
        rec,
        "free_rigid_*_friction = 0.01 / 0.004",
        free_rigid_torsional_friction=0.01,
        free_rigid_rolling_friction=0.004,
    )
    simulate(
        rec,
        "override 0.03 / 0.012 (3x)",
        free_rigid_torsional_friction=0.03,
        free_rigid_rolling_friction=0.012,
    )
    rec.save()
