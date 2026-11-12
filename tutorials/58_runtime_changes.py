"""Change model properties mid-run: edit friction in place, then call notify_model_changed."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
SLOPE = math.radians(25.0)  # tan(25 deg) = 0.47


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), SLOPE)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.8)
    builder.add_shape_box(
        -1, xform=wp.transform((0.0, 0.0, 0.4), tilt), hx=0.9, hy=0.4, hz=0.03, cfg=cfg
    )
    normal = np.array(wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, 1.0)))
    uphill = np.array(wp.quat_rotate(tilt, wp.vec3(1.0, 0.0, 0.0)))
    for y in (-0.15, 0.15):
        box = builder.add_body(
            xform=wp.transform(np.array([0.0, y, 0.4]) + 0.55 * uphill + 0.081 * normal, tilt)
        )
        builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    return builder.finalize()


def simulate(rec, label, notify):
    model = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    start = state_0.body_q.numpy()[:, :3].copy()

    run = rec.variant(label, model)
    for frame in range(rec.frames(3.0)):
        if frame == rec.fps:
            # at 1 s, make the second box slippery: shapes are ground, ramp, box 0, box 1
            mu = model.shape_material_mu.numpy()
            mu[3] = 0.05  # pair average 0.425 < tan(slope)
            model.shape_material_mu.assign(mu)
            if notify:
                solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        slide = np.linalg.norm(state_0.body_q.numpy()[:, :3] - start, axis=1)
        run.frame(state_0, unchanged_box=slide[0], edited_box=slide[1])
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "edited at 1 s, no notify", False)
    simulate(rec, "edited at 1 s + notify_model_changed", True)
    rec.save()
