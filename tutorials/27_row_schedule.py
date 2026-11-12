"""row_schedule: the order of contact rows and internal rows (drives, limits, mimics)."""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8
SCHEDULES = ["interleaved", "contact_then_internal", "phased"]
LIMIT = 0.05  # rad; the hinge stops this far below level


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build_stop():
    """A hinged arm resting on its lower limit, loaded by a heavy box near its tip."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    arm = builder.add_link(xform=tf(0.25, 0.0, 0.3))
    builder.add_shape_box(
        arm, hx=0.25, hy=0.05, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=300.0)
    )
    hinge = builder.add_joint_revolute(
        -1,
        arm,
        axis=(0.0, -1.0, 0.0),
        parent_xform=tf(0.0, 0.0, 0.3),
        child_xform=tf(-0.25, 0.0, 0.0),
        limit_lower=-LIMIT,
        limit_upper=0.5,
    )
    builder.add_articulation([hinge])
    box = builder.add_body(xform=tf(0.42, 0.0, 0.36))
    builder.add_shape_box(
        box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=8000.0)
    )
    return builder.finalize()


def build_gripper(mu=0.3):
    """Fingers squeezed by effort-limited drive rows lift and shake a box."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
    base = builder.add_link(xform=tf(0.0, 0.0, 0.45))
    builder.add_shape_box(
        base, hx=0.16, hy=0.03, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=500.0)
    )
    joints = [
        builder.add_joint_prismatic(
            -1,
            base,
            axis=newton.Axis.Z,
            parent_xform=tf(0.0, 0.0, 0.45),
            target_ke=4000.0,
            target_kd=200.0,
            armature=0.5,
        )
    ]
    finger_cfg = newton.ModelBuilder.ShapeConfig(density=800.0, mu=mu)
    for side in (-1.0, 1.0):
        finger = builder.add_link(xform=tf(0.12 * side, 0.0, 0.3))
        builder.add_shape_box(finger, hx=0.01, hy=0.03, hz=0.12, cfg=finger_cfg)
        joints.append(
            builder.add_joint_prismatic(
                base,
                finger,
                axis=newton.Axis.X,
                parent_xform=tf(0.12 * side, 0.0, -0.3),
                armature=0.05,
                limit_lower=0.0 if side < 0 else -0.12,
                limit_upper=0.12 if side < 0 else 0.0,
                target_ke=2000.0,
                target_kd=40.0,
                effort_limit=40.0,
            )
        )
    builder.add_articulation(joints)
    box = builder.add_body(xform=tf(0.0, 0.0, 0.0601))
    builder.add_shape_box(
        box, hx=0.05, hy=0.05, hz=0.06, cfg=newton.ModelBuilder.ShapeConfig(density=600.0, mu=mu)
    )
    return builder.finalize(), base, box


def simulate_stop(rec, schedule):
    model = build_stop()
    solver = SolverSteel(model, enable_joint_limits=True, iterations=4, row_schedule=schedule)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(f'stop, "{schedule}"', model, scene="stop")
    for _ in range(rec.frames(2.0)):
        angle = state_0.joint_q.numpy()[0]
        box_z = state_0.body_q.numpy()[1, 2]
        run.frame(state_0, past_limit_deg=math.degrees(max(0.0, -LIMIT - angle)), box_height=box_z)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


def simulate_grasp(rec, schedule):
    model, base, box = build_gripper()
    solver = SolverSteel(
        model,
        enable_joint_limits=True,
        articulation_constraint_max=128,
        iterations=4,
        drive_mode="rows",
        row_schedule=schedule,
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(f'grasp, "{schedule}"', model, scene="grasp")
    for frame in range(rec.frames(7.0)):
        t = frame / rec.fps
        q = state_0.body_q.numpy()
        run.frame(state_0, slip_mm=1000.0 * ((q[base, 2] - q[box, 2]) - (0.45 - 0.0601)))
        lift = min(0.25, max(0.0, 0.25 * (t - 1.0)))
        shake = 0.03 * math.sin(2.0 * math.pi * 3.0 * (t - 2.5)) if 2.5 < t < 5.5 else 0.0
        close = (
            0.1 if t > 0.3 else 0.0
        )  # targets past the box; the effort limit sets the grip force
        control.joint_target_q.assign(np.array([lift + shake, close, -close], dtype=np.float32))
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    for schedule in SCHEDULES:
        simulate_stop(rec, schedule)
    for schedule in SCHEDULES:
        simulate_grasp(rec, schedule)
    rec.save()
