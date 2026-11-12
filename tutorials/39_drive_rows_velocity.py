"""Drive rows in velocity iterations (velocity_drive_mode) and fused velocity limits."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build_pusher():
    """A velocity-driven pusher shoving a heavy box that starts 1.5 cm inside it."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    pusher = builder.add_link(xform=tf(0.0, 0.0, 0.05))
    builder.add_shape_box(
        pusher,
        hx=0.05,
        hy=0.05,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.0),
    )
    slide = builder.add_joint_prismatic(
        -1,
        pusher,
        axis=newton.Axis.X,
        parent_xform=tf(0.0, 0.0, 0.05),
        target_kd=80.0,
        target_qd=1.0,
        effort_limit=20.0,
    )
    builder.add_articulation([slide])
    box = builder.add_body(xform=tf(0.085, 0.0, 0.05))
    builder.add_shape_box(
        box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=2000.0, mu=0.5)
    )
    return builder.finalize()


def build_flywheel():
    """A flywheel whose velocity drive asks for 10 rad/s against a 6 rad/s limit."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    hub = builder.add_link(xform=tf(0.0, 0.0, 0.5))
    builder.add_shape_cylinder(
        hub, radius=0.2, half_height=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=500.0)
    )
    builder.add_shape_box(hub, xform=tf(0.15, 0.0, 0.03), hx=0.04, hy=0.02, hz=0.01)
    hinge = builder.add_joint_revolute(
        -1,
        hub,
        axis=newton.Axis.Z,
        parent_xform=tf(0.0, 0.0, 0.5),
        velocity_limit=6.0,
        target_kd=2.0,
        target_qd=10.0,
    )
    builder.add_articulation([hinge])
    return builder.finalize()


def simulate(rec, label, scene, **options):
    model = build_pusher() if scene == "pusher" else build_flywheel()
    solver = SolverSteel(model, drive_mode="rows", **options)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    run = rec.variant(label, model, scene=scene)
    fastest = 0.0  # the largest speed reached in any step of the frame
    for _ in range(rec.frames(1.0 if scene == "pusher" else 2.0)):
        if scene == "pusher":
            run.frame(state_0, pusher_speed=state_0.joint_qd.numpy()[0])
        else:
            run.frame(state_0, omega_max=fastest)
        fastest = 0.0
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            fastest = max(fastest, abs(state_0.joint_qd.numpy()[0]))
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    pusher = {"velocity_iterations": 2, "iterations": 3, "baumgarte": 0.8}
    simulate(rec, 'velocity_drive_mode="freeze" (default)', "pusher", **pusher)
    simulate(rec, 'velocity_drive_mode="active"', "pusher", velocity_drive_mode="active", **pusher)
    # rows are created only above 90% of the limit, so the unfused row engages a step late
    flywheel = {"enable_joint_velocity_limits": True, "velocity_limit_activation_fraction": 0.9}
    simulate(
        rec,
        "fuse_joint_velocity_limits=False",
        "flywheel",
        fuse_joint_velocity_limits=False,
        **flywheel,
    )
    simulate(rec, "fuse_joint_velocity_limits=True (default)", "flywheel", **flywheel)
    rec.save()
