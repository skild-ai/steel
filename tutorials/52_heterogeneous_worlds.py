"""Heterogeneous worlds: different structures in one model, stepped by one solver."""

import newton
import warp as wp
from newton.solvers import SolverSteel

from replay import Recorder

DT = 1.0 / 240.0
SUBSTEPS = 8


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def chain(builder, links, x):
    parent, joints = -1, []
    for i in range(links):
        link = builder.add_link(xform=tf(x, 0.0, 0.95 - 0.1 * i))
        builder.add_shape_capsule(
            link,
            radius=0.015,
            half_height=0.035,
            cfg=newton.ModelBuilder.ShapeConfig(density=800.0),
        )
        axis = newton.Axis.Y if i % 2 == 0 else newton.Axis.X
        anchor = tf(x, 0.0, 1.0) if parent < 0 else tf(0.0, 0.0, -0.05)
        # a little armature keeps long chains of very light links stable
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=axis,
                parent_xform=anchor,
                child_xform=tf(0.0, 0.0, 0.05),
                armature=0.001,
            )
        )
        builder.joint_q[-1] = 0.5  # start every hinge at 0.5 rad
        parent = link
    builder.add_articulation(joints)


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for w, kind in enumerate(["2-link chain", "5-link chain", "9-link chain", "pile", "arm + box"]):
        x = 0.6 * w - 1.2
        builder.begin_world(label=kind)
        if kind.endswith("chain"):
            chain(builder, int(kind[0]), x)
        elif kind == "pile":
            for i in range(12):
                box = builder.add_body(
                    xform=tf(x + 0.03 * (i % 3) - 0.03, 0.02 * (i % 2), 0.06 + 0.09 * i)
                )
                builder.add_shape_box(box, hx=0.04, hy=0.04, hz=0.04)
        else:
            arm = builder.add_link(xform=tf(x, 0.0, 0.3))
            builder.add_shape_box(
                arm, hx=0.02, hy=0.02, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=500.0)
            )
            j = builder.add_joint_revolute(
                -1,
                arm,
                axis=newton.Axis.Y,
                parent_xform=tf(x, 0.0, 0.5),
                child_xform=tf(0.0, 0.0, 0.2),
                target_ke=300.0,
                target_kd=10.0,
                target_q=0.6,
            )
            builder.add_articulation([j])
            box = builder.add_body(xform=tf(x + 0.15, 0.0, 0.05))
            builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
        builder.end_world()
    return builder.finalize()


def simulate(rec, label):
    model = build()
    solver = SolverSteel(model, articulation_constraint_max=96, free_body_constraint_max=2048)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    for _ in range(rec.frames(4.0)):
        run.frame(state_0, worlds=model.world_count)
        for _ in range(SUBSTEPS):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
    run.check(solver)


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "one SolverSteel, five worlds")
    rec.save()
