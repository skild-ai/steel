"""Tethers: Steel ignores DISTANCE joint limits; a ball joint plus a limited slider holds them."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
ANCHOR = (0.0, 0.0, 1.0)
LENGTH = 0.4  # m, the most the weight may hang from the anchor


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build(kind):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_shape_sphere(
        -1,
        xform=tf(*ANCHOR),
        radius=0.02,
        cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False),
    )
    start = (0.3, 0.0, ANCHOR[2] - 0.1)  # 0.32 m from the anchor, swinging out
    if kind == "distance":
        weight = builder.add_link(xform=tf(*start))
        builder.add_shape_sphere(weight, radius=0.05)
        # a DISTANCE root joint acts as a free joint in Steel: max_distance is not enforced
        joints = [
            builder.add_joint_distance(-1, weight, parent_xform=tf(*ANCHOR), max_distance=LENGTH)
        ]
        # add_joint_distance leaves joint_q at zero, which would put the weight on the anchor
        builder.joint_q[-7:-4] = [start[i] - ANCHOR[i] for i in range(3)]
    else:
        # a massless swivel at the anchor, then a slider out along the tether limited to [0, LENGTH]
        swivel = builder.add_link(xform=tf(*ANCHOR), mass=0.0)
        weight = builder.add_link(xform=tf(*ANCHOR))
        builder.add_shape_sphere(weight, radius=0.05)
        joints = [
            builder.add_joint_ball(-1, swivel, parent_xform=tf(*ANCHOR), armature=1e-4),
            builder.add_joint_prismatic(
                swivel,
                weight,
                axis=newton.Axis.Z,
                limit_lower=-LENGTH,
                limit_upper=0.0,
                armature=1e-4,
            ),
        ]
        # point the slider at the start position: tilt the swivel, extend the slider
        tilt = math.atan2(start[0], ANCHOR[2] - start[2])
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -tilt)
        builder.joint_q[-5:-1] = [rot[0], rot[1], rot[2], rot[3]]
        builder.joint_q[-1] = -math.hypot(start[0], ANCHOR[2] - start[2])
    builder.add_articulation(joints)
    return builder.finalize(), weight


def simulate(rec, label, kind):
    model, weight = build(kind)
    solver = SolverSteel(model, enable_joint_limits=True)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(3.0)):
        p = state_0.body_q.numpy()[weight, :3]
        run.frame(state_0, distance_m=np.linalg.norm(p - np.array(ANCHOR)), limit_m=LENGTH)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "DISTANCE joint, max 0.4 m (ignored)", "distance")
    simulate(rec, "ball + prismatic in [0, 0.4] m", "tether")
    rec.save()
