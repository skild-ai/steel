"""Friction combine modes: how two shapes' mu values become one contact coefficient."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
SLOPE = math.radians(25.0)  # tan(25 deg) = 0.47
BOX_MU = [0.1, 0.4, 0.9]
RAMP_MU = 0.4


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), SLOPE)
    builder.add_shape_box(
        -1,
        xform=wp.transform((0.0, 0.0, 0.4), tilt),
        hx=0.9,
        hy=0.45,
        hz=0.03,
        cfg=newton.ModelBuilder.ShapeConfig(mu=RAMP_MU),
    )
    normal = np.array(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, 1.0)))
    uphill = np.array(wp.quat_rotate(tilt, wp.vec3(1.0, 0.0, 0.0)))
    for i, mu in enumerate(BOX_MU):
        p = np.array([0.0, 0.3 * i - 0.3, 0.4]) + 0.5 * uphill + 0.071 * normal
        box = builder.add_body(xform=wp.transform(p, tilt))
        builder.add_shape_box(
            box, hx=0.04, hy=0.04, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(mu=mu)
        )
    return builder.finalize()


def simulate(rec, mode):
    model = build()
    solver = SolverSteel(model, friction_combine=mode)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    start = state_0.body_q.numpy()[:, :3].copy()

    run = rec.variant(f'friction_combine="{mode}"', model)
    for _ in range(rec.frames(2.0)):
        slide = np.linalg.norm(state_0.body_q.numpy()[:, :3] - start, axis=1)
        run.frame(state_0, **{f"mu_{mu}": d for mu, d in zip(BOX_MU, slide)})
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    for mode in ["average", "min", "multiply", "max"]:
        simulate(rec, mode)
    rec.save()
