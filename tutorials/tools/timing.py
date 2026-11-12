"""Per-step cost of Steel options on small replicated scenes, timed by CUDA graph replay.

Each group builds every option at every world count, then times them in interleaved rounds and
keeps the median, so drift on a shared GPU affects all options alike.

usage: python tutorials/tools/timing.py OUT.json [group ...]
"""

import json
import sys
import time

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverSteel

FRAME_DT = 1.0 / 30.0


def tf(x, y, z):
    return wp.transform((x, y, z), wp.quat_identity())


def stack(n=8, h=0.05):
    b = newton.ModelBuilder()
    for i in range(n):
        body = b.add_body(xform=tf(0.0, 0.0, h + 2 * h * i + 0.0002 * i))
        b.add_shape_box(body, hx=h, hy=h, hz=h, cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
    return b


def chain(links=6, armature=0.001):
    b = newton.ModelBuilder()
    parent, joints = -1, []
    for i in range(links):
        link = b.add_link(xform=tf(0.0, 0.0, 1.0 - 0.1 * i - 0.05))
        b.add_shape_capsule(link, radius=0.02, half_height=0.03)
        anchor = tf(0.0, 0.0, 1.0) if parent < 0 else tf(0.0, 0.0, -0.05)
        joints.append(
            b.add_joint_revolute(
                parent,
                link,
                axis=newton.Axis.Y,
                parent_xform=anchor,
                target_ke=50.0,
                target_kd=1.0,
                child_xform=tf(0.0, 0.0, 0.05),
                armature=armature,
                friction=0.01,
            )
        )
        b.joint_q[-1] = 1.2 if i == 0 else 0.0
        parent = link
    b.add_articulation(joints)
    return b


def towers():
    b = newton.ModelBuilder()
    for k in range(4):
        for i in range(5):
            body = b.add_body(xform=tf(0.5 * k - 0.75, 0.0, 0.05 + 0.1005 * i))
            b.add_shape_box(
                body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=0.7)
            )
    return b


def arm_boxes():
    b = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
    base = b.add_link(xform=tf(0.0, -0.45, 0.06))
    b.add_shape_box(base, hx=0.04, hy=0.04, hz=0.04, cfg=cfg)
    joints = [
        b.add_joint_revolute(
            -1,
            base,
            axis=newton.Axis.Z,
            parent_xform=tf(0.0, -0.45, 0.06),
            target_kd=50.0,
            target_qd=-0.8,
        )
    ]
    parent = base
    for i in range(3):
        link = b.add_link(xform=tf(0.0, -0.45 + 0.12 + 0.24 * i, 0.06))
        b.add_shape_box(link, hx=0.02, hy=0.11, hz=0.03, cfg=cfg)
        anchor = (0.0, 0.0, 0.0) if i == 0 else (0.0, 0.12, 0.0)
        joints.append(
            b.add_joint_revolute(
                parent,
                link,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(anchor, wp.quat_identity()),
                child_xform=tf(0.0, -0.12, 0.0),
                target_ke=60.0,
                target_kd=2.0,
            )
        )
        parent = link
    b.add_articulation(joints)
    for i in range(5):
        for j in range(5):
            box = b.add_body(xform=tf(0.13 * i + 0.1, 0.13 * j - 0.36, 0.04))
            b.add_shape_box(
                box,
                hx=0.04,
                hy=0.04,
                hz=0.04,
                cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.6),
            )
    return b


def world(scene, worlds):
    b = newton.ModelBuilder()
    b.add_ground_plane()
    b.replicate(scene, worlds, spacing=(2.0, 2.0, 0.0))
    return b.finalize()


class Case:
    """One option at one world count, warmed up and captured once."""

    def __init__(
        self, label, scene, worlds, opts, pipe, substeps, warm_frames, eager, graph_frames
    ):
        self.label, self.worlds, self.substeps, self.eager = label, worlds, substeps, eager
        self.dt = FRAME_DT / substeps  # every option simulates the same 30 Hz frame
        self.graph_frames = graph_frames
        model = world(scene(), worlds)
        self.pipeline = newton.CollisionPipeline(model, **pipe)
        self.contacts = self.pipeline.contacts()
        self.s0, self.s1, self.control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.s0)
        wp.synchronize()
        mem0 = wp.get_mempool_used_mem_current(wp.get_device())
        self.solver = SolverSteel(model, **opts)
        wp.synchronize()
        # solver allocations at construction only; warm-up and graph capture are not counted
        self.mem_mb = (wp.get_mempool_used_mem_current(wp.get_device()) - mem0) / 1e6
        for _ in range(warm_frames):
            self.frame()
        wp.synchronize()
        self.graph = None
        if not eager:
            # capture whole refresh cycles, e.g. 16 steps for update_mass_matrix_interval=16
            with wp.ScopedCapture() as cap:
                for _ in range(graph_frames):
                    self.frame()
            self.graph = cap.graph
        self.samples = []

    def frame(self):
        for _ in range(self.substeps):
            self.pipeline.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
            self.s0, self.s1 = self.s1, self.s0

    def time(self, frames):
        wp.synchronize()
        start = time.perf_counter()
        launches = frames // self.graph_frames if self.graph is not None else frames
        for _ in range(launches):
            wp.capture_launch(self.graph) if self.graph is not None else self.frame()
        wp.synchronize()
        steps = launches * (self.graph_frames if self.graph is not None else 1) * self.substeps
        self.samples.append(1000.0 * (time.perf_counter() - start) / steps)


def group(
    name,
    scene,
    options,
    world_counts=(1, 64, 1024),
    substeps=8,
    warm_frames=60,
    rounds=7,
    frames=20,
    graph_frames=1,
):
    """options: list of (label, solver opts, pipeline opts, substeps or None, eager)."""
    rows = []
    for worlds in world_counts:
        cases = [
            Case(
                label, scene, worlds, opts, pipe, sub or substeps, warm_frames, eager, graph_frames
            )
            for label, opts, pipe, sub, eager in options
        ]
        for _ in range(rounds):
            for case in cases:
                case.time(frames)
        for case in cases:
            try:
                case.solver.check_constraint_capacity()
                ok = True
            except RuntimeError:
                ok = False
            ms = float(np.median(case.samples))
            row = {
                "group": name,
                "label": case.label,
                "worlds": worlds,
                "substeps": case.substeps,
                "ms_per_step": round(ms, 4),
                "ms_per_frame": round(ms * case.substeps, 4),
                "spread": round(float(np.max(case.samples) - np.min(case.samples)), 4),
                "solver_mem_mb": round(case.mem_mb, 2),
                "rows_ok": ok,
            }
            print(json.dumps(row), flush=True)
            rows.append(row)
        del cases
    return rows


def o(label, opts=None, pipe=None, substeps=None, eager=False):
    return (label, dict(opts or {}), dict(pipe or {}), substeps, eager)


CAPS = {"articulation_constraint_max": 384, "free_body_constraint_max": 2048}
GROUPS = {
    "patch": (
        stack,
        [o("patch friction (default)"), o("point friction", {"friction_anchor_beta": 0.0})],
    ),
    "velocity_iterations": (
        stack,
        [o("velocity_iterations=0"), o("velocity_iterations=4", {"velocity_iterations": 4})],
    ),
    "nesterov": (
        stack,
        [
            o("iterations=6", {"iterations": 6}),
            o("iterations=6 + Nesterov", {"iterations": 6, "nesterov_momentum": True}),
            o("iterations=12"),
            o("iterations=12 + Nesterov", {"nesterov_momentum": True}),
            o("iterations=24", {"iterations": 24}),
        ],
    ),
    "substeps": (
        stack,
        [
            o(f"{s} x {i}", {"iterations": i}, substeps=s)
            for s, i in ((2, 24), (4, 12), (8, 6), (16, 3))
        ],
    ),
    "warmstart": (
        stack,
        [
            o("iterations=4", {"iterations": 4}),
            o(
                "iterations=4 + warm start",
                {"iterations": 4, "contact_warmstart": True},
                {"contact_matching": "latest"},
            ),
            o(
                "iterations=8 + warm start",
                {"iterations": 8, "contact_warmstart": True},
                {"contact_matching": "latest"},
            ),
            o("iterations=12"),
        ],
    ),
    "regularization": (
        stack,
        [
            o("contact_regularization=0"),
            o("contact_regularization=0.02", {"contact_regularization": 0.02}),
        ],
    ),
    "determinism": (
        stack,
        [
            o("default"),
            o(
                "RUN_TO_RUN",
                {"deterministic": wp.DeterministicMode.RUN_TO_RUN},
                {"deterministic": True},
            ),
        ],
    ),
    "graph": (stack, [o("eager", eager=True), o("CUDA graph")]),
    "capacity": (
        stack,
        [
            o("free_body_constraint_max=128", {"free_body_constraint_max": 128}),
            o("free_body_constraint_max=512 (default)"),
            o("free_body_constraint_max=2048", {"free_body_constraint_max": 2048}),
        ],
    ),
    "drives": (
        chain,
        [o('drive_mode="augmented"'), o('drive_mode="rows"', {"drive_mode": "rows"})],
    ),
    "joint_friction": (
        chain,
        [o("joint friction off"), o("enable_joint_friction=True", {"enable_joint_friction": True})],
    ),
    "interval": (
        chain,
        [
            o(f"update_mass_matrix_interval={k}", {"update_mass_matrix_interval": k})
            for k in (1, 4, 16)
        ],
    ),
    "sleeping": (
        towers,
        [o("enable_sleeping=False"), o("enable_sleeping=True", {"enable_sleeping": True})],
    ),
    "colored": (
        arm_boxes,
        [
            o("serial + immediate", CAPS),
            o(
                "colored + propagation",
                dict(CAPS, solve_mode="colored", articulated_contact_response="propagation"),
            ),
        ],
    ),
}


def main():
    names = sys.argv[2:] or list(GROUPS)
    rows = []
    for name in names:
        scene, options = GROUPS[name]
        kwargs = {"warm_frames": 120} if name == "sleeping" else {}
        if name == "colored":
            kwargs["world_counts"] = (1, 64, 256)
        if name == "interval":
            kwargs["graph_frames"] = 2
        rows += group(name, scene, options, **kwargs)
    json.dump({"gpu": wp.get_device().name, "rows": rows}, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
