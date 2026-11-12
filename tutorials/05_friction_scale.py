"""contact_friction_scale: scale every shape's friction in the solver without editing the model."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
SLOPE = math.radians(20.0)  # tan(20 deg) = 0.36
MU = 0.6


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), SLOPE)
    cfg = newton.ModelBuilder.ShapeConfig(mu=MU)
    builder.add_shape_box(
        -1, xform=wp.transform((0.0, 0.0, 0.4), tilt), hx=0.9, hy=0.2, hz=0.03, cfg=cfg
    )
    normal = np.array(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, 1.0)))
    uphill = np.array(wp.quat_rotate(tilt, wp.vec3(1.0, 0.0, 0.0)))
    box = builder.add_body(
        xform=wp.transform(np.array([0.0, 0.0, 0.4]) + 0.55 * uphill + 0.081 * normal, tilt)
    )
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    return builder.finalize()


def simulate(rec, scale):
    model = build()
    solver = SolverSteel(model, contact_friction_scale=scale)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    start = state_0.body_q.numpy()[0, :3].copy()

    run = rec.variant(f"contact_friction_scale={scale}  (effective mu {MU * scale:.2f})", model)
    for _ in range(rec.frames(2.0)):
        run.frame(state_0, slide_m=np.linalg.norm(state_0.body_q.numpy()[0, :3] - start))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    for scale in [1.0, 0.7, 0.4]:
        simulate(rec, scale)
    rec.save()
