"""Mimic joints as gears: q_follower = offset + ratio * q_leader, editable at runtime."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
UPRIGHT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2)  # cylinder axis along y


def tf(x, y, z, q=None):
    return wp.transform((x, y, z), q if q is not None else wp.quat_identity())


def wheel(builder, base, x, radius, **joint):
    """A disc on a hinge about y with a spoke, so its rotation is visible."""
    link = builder.add_link(xform=tf(x, 0.0, 0.5))
    builder.add_shape_cylinder(
        link,
        xform=tf(0.0, 0.0, 0.0, UPRIGHT),
        radius=radius,
        half_height=0.015,
        cfg=newton.ModelBuilder.ShapeConfig(density=300.0),
    )
    builder.add_shape_box(
        link, xform=tf(0.0, -0.025, radius * 0.5), hx=0.012, hy=0.006, hz=radius * 0.5
    )
    return builder.add_joint_revolute(
        base, link, axis=newton.Axis.Y, parent_xform=tf(x, 0.0, 0.5), **joint
    )


def build():
    builder = newton.ModelBuilder()
    base = builder.add_link()
    root = builder.add_joint_fixed(-1, base)
    # the leader is driven at 1 rad/s; the follower meshes with it at a 2:1 ratio
    leader = wheel(builder, base, -0.3, 0.2, target_kd=50.0, target_qd=1.0)
    follower = wheel(builder, base, 0.02, 0.1)
    builder.add_articulation([root, leader, follower])
    builder.set_joint_mimic(follower, leader, coeffs=(0.0, -2.0))
    return builder.finalize(), follower


def simulate(rec, label):
    model, follower = build()
    solver = SolverSteel(model)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model)
    for frame in range(rec.frames(6.0)):
        if frame == rec.fps * 3:
            # switch gears at 3 s; the offset keeps the follower in place.
            # The solver reads the coefficients every step.
            leader_q = state_0.joint_q.numpy()[0]
            coeffs = model.joint_mimic_coeffs.numpy()
            coeffs[follower] = (-1.5 * leader_q, -0.5)
            model.joint_mimic_coeffs.assign(coeffs)
        q = np.degrees(state_0.joint_q.numpy())
        run.frame(state_0, leader_deg=q[0], follower_deg=q[1])
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "ratio -2, switched to -0.5 at 3 s")
    rec.save()
