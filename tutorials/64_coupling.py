# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Coupling: Steel simulates a chain, boxes and a stack; XPBD the cloth they land on."""

import time

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupledProxy

from replay import Recorder

DT = 1.0 / 960.0
SUBSTEPS = 32  # 30 Hz frames
HX = 0.06  # half length of a chain link


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def build():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    cfg = newton.ModelBuilder.ShapeConfig(density=100.0)
    for x, z in ((0.2, 0.9), (-0.15, 1.1)):
        box = builder.add_body(xform=tf(x, 0.0, z))
        builder.add_shape_box(box, hx=0.08, hy=0.08, hz=0.08, cfg=cfg)
    # a free-floating six-link chain: its joints are tree joints in Steel, constraints in XPBD
    link = builder.add_link(xform=tf(-0.3, 0.0, 1.4))
    builder.add_shape_box(link, hx=HX, hy=0.03, hz=0.03, cfg=cfg)
    joints = [builder.add_joint_free(link)]
    for _ in range(5):
        child = builder.add_link()
        builder.add_shape_box(child, hx=HX, hy=0.03, hz=0.03, cfg=cfg)
        joints.append(
            builder.add_joint_revolute(
                link,
                child,
                axis=newton.Axis.Y,
                target_kd=0.5,
                parent_xform=tf(HX, 0.0, 0.0),
                child_xform=tf(-HX, 0.0, 0.0),
            )
        )
        link = child
    builder.add_articulation(joints)
    # a stack of six 6 cm boxes on the floor beside the cloth
    for i in range(6):
        box = builder.add_body(xform=tf(0.0, 0.8, 0.03 + 0.0602 * i))
        builder.add_shape_box(box, hx=0.03, hy=0.03, hz=0.03, cfg=cfg)
    rigid_bodies = list(range(builder.body_count))
    # a 1 m cloth fixed along two edges, 0.5 m above the ground
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5, -0.5, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=16,
        dim_y=16,
        cell_x=1.0 / 16,
        cell_y=1.0 / 16,
        mass=0.1,
        fix_left=True,
        fix_right=True,
        add_springs=True,
        spring_ke=1.0e3,
        spring_kd=1.0,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=1.0e-1,
        edge_ke=1.0e2,
        edge_kd=1.0e-1,
        particle_radius=0.01,
    )
    model = builder.finalize()
    model.soft_contact_ke = 1.0e2
    model.soft_contact_kd = 1.0
    return model, rigid_bodies


def make_solver(model, rigid_bodies, coupled):
    if not coupled:
        return SolverXPBD(model, iterations=40)
    return SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupledProxy.Entry(
                name="steel",
                solver=lambda view: SolverSteel(view, articulation_constraint_max=256),
                bodies=rigid_bodies,
                joints=list(range(model.joint_count)),
            ),
            SolverCoupledProxy.Entry(
                name="xpbd",
                solver=lambda view: SolverXPBD(view, iterations=40),
                particles=list(range(model.particle_count)),
            ),
        ],
        # XPBD sees each Steel body as a proxy with Steel's effective mass; impulses flow back
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(source="steel", destination="xpbd", bodies=rigid_bodies)
            ]
        ),
    )


def joint_gap_mm(body_q):
    """Largest distance between the two anchors of a chain joint [mm]."""
    gaps = []
    for parent in range(2, 7):
        a, b = body_q[parent], body_q[parent + 1]
        pa = wp.transform_point(
            wp.transform(wp.vec3(*a[:3]), wp.quat(*a[3:])), wp.vec3(HX, 0.0, 0.0)
        )
        pb = wp.transform_point(
            wp.transform(wp.vec3(*b[:3]), wp.quat(*b[3:])), wp.vec3(-HX, 0.0, 0.0)
        )
        gaps.append(np.linalg.norm(np.array(pa) - np.array(pb)))
    return 1000.0 * max(gaps)


def simulate(rec, label, coupled):
    model, rigid_bodies = build()
    solver = make_solver(model, rigid_bodies, coupled)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    run = rec.variant(label, model)
    elapsed = 0.0
    for frame in range(rec.frames(4.0)):
        q = state_0.body_q.numpy()
        cloth_z = state_0.particle_q.numpy()[:, 2].min()
        stack_sag_mm = 1000.0 * (0.03 + 0.06 * 5 - q[-1, 2])
        run.frame(
            state_0,
            particles=state_0.particle_q,
            joint_gap_mm=joint_gap_mm(q),
            cloth_low_z=cloth_z,
            stack_sag_mm=stack_sag_mm,
        )
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(SUBSTEPS):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        if frame > 0:  # the first frame includes kernel compilation
            elapsed += time.perf_counter() - start
    if coupled:
        run.check(solver.solver("steel"))
    print(f"{label}: {1000.0 * elapsed / ((rec.frames(4.0) - 1) * SUBSTEPS):.3f} ms per eager step")


if __name__ == "__main__":
    rec = Recorder()
    simulate(rec, "XPBD for everything", False)
    simulate(rec, "Steel rigid + XPBD cloth, coupled", True)
    rec.save()
