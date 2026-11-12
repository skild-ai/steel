"""joint_limit_activation_gap: create joint-limit rows only near a bound."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
LINKS = 12
LIMIT = math.radians(20.0)


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    """A twelve-link chain with +-20 degree hinges whose joints slam into their limits."""
    builder = newton.ModelBuilder()
    parent, joints = -1, []
    for i in range(LINKS):
        link = builder.add_link(xform=tf(0.0, 0.0, 1.0 - 0.12 * i - 0.06))
        # no collisions: every row in this scene is a joint-limit row
        cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
        builder.add_shape_capsule(link, radius=0.025, half_height=0.035, cfg=cfg)
        builder.add_shape_sphere(
            link,
            xform=tf(0.0, 0.0, -0.05),
            radius=0.03,
            cfg=newton.ModelBuilder.ShapeConfig(density=4000.0, has_shape_collision=False),
        )
        anchor = tf(0.0, 0.0, 1.0) if parent < 0 else tf(0.0, 0.0, -0.06)
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=newton.Axis.Y,
                parent_xform=anchor,
                child_xform=tf(0.0, 0.0, 0.06),
                limit_lower=-LIMIT,
                limit_upper=LIMIT,
                armature=0.001,
            )
        )
        builder.joint_q[-1] = 1.4 if i == 0 else 0.0  # the root starts far outside its limit
        parent = link
    builder.add_articulation(joints)
    return builder.finalize()


def simulate(rec, label, gap):
    model = build()
    solver = SolverSteel(
        model,
        enable_joint_limits=True,
        joint_limit_activation_gap=gap,
        row_watermark=True,
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    worst = 0.0
    for _ in range(rec.frames(3.0)):
        # limit violation of the lower joints, which start inside their limits
        q = np.abs(state_0.joint_q.numpy()[1:])
        worst = max(worst, math.degrees(max(0.0, q.max() - LIMIT)))
        rows = solver.constraint_row_watermarks()["articulation_raw_high_water"]
        run.frame(
            state_0,
            violation_deg=math.degrees(max(0.0, q.max() - LIMIT)),
            limit_rows_high_water=rows,
        )
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)
    print(f"{label}: worst violation {worst:.2f} deg")


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "gap=None (default, speed-aware)", None)
    simulate(rec, "gap=inf (every finite limit, every step)", float("inf"))
    simulate(rec, "gap=0 (rows only past the bound)", 0.0)
    rec.save()
