# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0

"""Per-step cost of the limit-gap, performance, friction, propagation and RL tutorials' options.

Timed like timing.py. Scenes come from the tutorial scripts; each is replicated into 1, 64
and 1024 worlds.

usage: python tutorials/tools/timing_planned.py OUT.json [group ...]
"""

import importlib.util
import json
import os
import sys
import time

import newton
import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
TUTORIALS = os.path.dirname(HERE)
GROUP_NAMES = ["limit_gap", "performance", "friction_iterations", "propagation", "rl"]
sys.path.insert(0, TUTORIALS)
sys.path.insert(0, HERE)
import timing  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(TUTORIALS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_of(module, fn="build"):
    """The builder a tutorial's build function finalizes, so it can be replicated."""

    def make():
        finalize = newton.ModelBuilder.finalize
        newton.ModelBuilder.finalize = lambda self, *args, **kwargs: self
        try:
            return getattr(module, fn)()
        finally:
            newton.ModelBuilder.finalize = finalize

    return make


def world(scene, worlds):
    # each tutorial scene brings its own ground, so no extra ground here
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(3.0, 3.0, 0.0))
    return builder.finalize()


timing.world = world
o = timing.o


def groups():
    limit = load("38_limit_activation_gap")
    perf = load("60_performance_options")
    friction = load("12_friction_iterations")
    prop = load("61_propagation_options")
    limit_caps = {"enable_joint_limits": True}
    prop_base = dict(prop.BASE, articulation_constraint_max=256, free_body_constraint_max=256)
    return {
        "limit_gap": (
            scene_of(limit),
            [
                o("gap=None (default)", limit_caps),
                o("gap=inf", dict(limit_caps, joint_limit_activation_gap=float("inf"))),
                o("gap=0", dict(limit_caps, joint_limit_activation_gap=0.0)),
            ],
        ),
        "performance": (
            scene_of(perf),
            [
                o("defaults", {"articulation_constraint_max": 256}),
                o("parallel_tree", {"articulation_constraint_max": 256, "parallel_tree": True}),
                o(
                    "use_parallel_streams",
                    {"articulation_constraint_max": 256, "use_parallel_streams": True},
                ),
                o("double_buffer", {"articulation_constraint_max": 256, "double_buffer": True}),
                o(
                    "reuse_kinematics",
                    {"articulation_constraint_max": 256, "reuse_kinematics": True},
                ),
            ],
        ),
        "friction_iterations": (
            scene_of(friction, "build_pile"),
            [
                o("default", {"free_body_constraint_max": 2048}),
                o(
                    "contact_friction_position_iterations=4",
                    {"free_body_constraint_max": 2048, "contact_friction_position_iterations": 4},
                ),
                o(
                    "contact_friction_gap_max=0",
                    {"free_body_constraint_max": 2048, "contact_friction_gap_max": 0.0},
                ),
            ],
        ),
        "propagation": (
            scene_of(prop),
            [
                o(
                    "serial + immediate",
                    {"articulation_constraint_max": 256, "free_body_constraint_max": 256},
                ),
                o("colored + propagation (defaults)", prop_base),
                o(
                    "propagation_same_articulation_rows=True",
                    dict(prop_base, propagation_same_articulation_rows=True),
                ),
                o(
                    "propagation_cached_response=False",
                    dict(prop_base, propagation_cached_response=False),
                ),
                o(
                    'propagation_color_order="rotating"',
                    dict(prop_base, propagation_color_order="rotating"),
                ),
                o(
                    "propagation_constraint_max=64, free_body_constraint_max=8",
                    dict(prop_base, free_body_constraint_max=8, propagation_constraint_max=64),
                ),
            ],
        ),
    }


def rl_rows(world_counts=(16, 256, 1024, 4096, 16384), rounds=7, frames=20):
    """Time per frame of the RL loop (eight steps plus done and reset kernels, one graph launch)."""
    rl = load("62_rl_setup")
    measured = {}
    solver_class = rl.SolverSteel

    def measured_solver(model, **options):
        # solver allocations at construction only, as in timing.py
        wp.synchronize()
        mem0 = wp.get_mempool_used_mem_current(wp.get_device())
        solver = solver_class(model, **options)
        wp.synchronize()
        measured["mb"] = (wp.get_mempool_used_mem_current(wp.get_device()) - mem0) / 1e6
        return solver

    rl.SolverSteel = measured_solver
    envs = {}
    for worlds in world_counts:
        env = rl.Env(worlds)
        env.mem_mb = measured["mb"]
        for _ in range(3):
            env.run_frame()
        env.samples = []
        envs[worlds] = env
    for _ in range(rounds):
        for env in envs.values():
            wp.synchronize()
            start = time.perf_counter()
            for _ in range(frames):
                env.run_frame()
            wp.synchronize()
            env.samples.append(1000.0 * (time.perf_counter() - start) / (frames * rl.SUBSTEPS))
    rows = []
    for worlds, env in envs.items():
        ms = float(np.median(env.samples))
        row = {
            "group": "rl",
            "label": "frame graph: 8 steps, done and reset",
            "worlds": worlds,
            "substeps": rl.SUBSTEPS,
            "ms_per_step": round(ms, 4),
            "ms_per_frame": round(ms * rl.SUBSTEPS, 4),
            "us_per_world_step": round(1000.0 * ms / worlds, 4),
            "spread": round(float(np.max(env.samples) - np.min(env.samples)), 4),
            "solver_mem_mb": round(env.mem_mb, 2),
            "rows_ok": True,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main():
    names = sys.argv[2:] or GROUP_NAMES
    table = groups()
    rows = []
    for name in names:
        if name == "rl":
            rows += rl_rows()
            continue
        scene, options = table[name]
        rows += timing.group(name, scene, options)
    json.dump({"gpu": wp.get_device().name, "rows": rows}, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
