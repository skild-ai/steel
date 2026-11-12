# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Trebuchet
#
# Trebuchet: a hinged arm with a hung counterweight throws a boulder from a cup at a dry brick wall.
#
# Ported from steel-webgpu's tools/scenes.py (trebuchet); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run trebuchet
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from common.steel_scenes import BLUE, GOLD, GREY, RED, TEAL, SceneExample, cfg, tf


class Example(SceneExample):
    file = __file__
    camera = {'eye': (6.0, -30.0, 10.0), 'target': (16.0, 0.0, 4.0)}

    def build(self):
        """Trebuchet: a hinged arm with a hung counterweight throws a boulder from a cup at a dry brick wall."""
        b = newton.ModelBuilder()
        b.add_ground_plane(cfg=cfg(mu=0.9))
        pivot_z, short_arm, long_arm, cock = 4.2, 1.5, 5.2, 0.7  # avbd cocks to 0.98 rad, which buries the cup
        frame = [b.add_shape_box(-1, xform=tf((0, sy * 1.5, pivot_z / 2)), hx=0.25, hy=0.25, hz=pivot_z / 2, cfg=cfg(mu=0.8), color=GREY) for sy in (-1, 1)]
        frame.append(b.add_shape_box(-1, xform=tf((0, 0, pivot_z)), hx=0.3, hy=1.2, hz=0.3, cfg=cfg(mu=0.5), color=GREY))
        q_arm = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -cock)
        off = (long_arm - short_arm) / 2
        pivot = wp.vec3(0, 0, pivot_z)
        arm_c = pivot + wp.quat_rotate(q_arm, wp.vec3(-off, 0, 0))
        arm = b.add_link(xform=wp.transform(arm_c, q_arm), label="arm")
        b.add_shape_box(arm, hx=(long_arm + short_arm) / 2, hy=0.2, hz=0.175, cfg=cfg(density=2500, mu=0.5), color=(0.55, 0.38, 0.22))
        # The cup is welded to the arm: its parts are arm shapes in the arm frame.
        tip = wp.vec3(-long_arm + off, 0, 0)
        for size, p in (((1.5, 1.3, 0.12), (0, 0, 0.24)), ((0.14, 1.3, 0.6), (0.72, 0, 0.55)), ((0.14, 1.3, 0.42), (-0.72, 0, 0.46)),
                        ((1.5, 0.12, 0.5), (0, 0.66, 0.5)), ((1.5, 0.12, 0.5), (0, -0.66, 0.5))):
            b.add_shape_box(arm, xform=tf(tip + wp.vec3(*p)), hx=size[0] / 2, hy=size[1] / 2, hz=size[2] / 2, cfg=cfg(density=600, mu=0.5), color=(0.45, 0.30, 0.18))
        # The arm turns inside the axle housing, as a joint's bodies do not collide.
        for a in [i for i, owner in enumerate(b.shape_body) if owner == arm]:
            for f in frame:
                b.add_shape_collision_filter_pair(a, f)
        j_arm = b.add_joint_revolute(-1, arm, axis=(0, 1, 0), parent_xform=wp.transform(pivot, q_arm), child_xform=tf((off, 0, 0)))
        anchor = wp.vec3(short_arm + off, 0, 0)
        cw_pos = pivot + wp.quat_rotate(q_arm, wp.vec3(short_arm, 0, 0)) + wp.vec3(0, 0, -1.15)
        cw = b.add_link(xform=tf(cw_pos), label="counterweight")
        b.add_shape_box(cw, hx=0.85, hy=0.85, hz=0.75, cfg=cfg(density=10000, mu=0.6), color=GREY)
        j_cw = b.add_joint_ball(arm, cw, parent_xform=wp.transform(anchor, wp.quat_inverse(q_arm)), child_xform=tf((0, 0, 1.15)))
        b.add_articulation([j_arm, j_cw])
        up = wp.quat_rotate(q_arm, wp.vec3(0, 0, 1))
        boulder = b.add_body(xform=tf(pivot + wp.quat_rotate(q_arm, wp.vec3(-long_arm, 0, 0)) + up * 0.85))
        b.add_shape_sphere(boulder, radius=0.55, cfg=cfg(density=3000, mu=0.4), color=(0.35, 0.33, 0.30))
        b.add_shape_box(-1, xform=tf((-1.9, 0, pivot_z + 2.9)), hx=0.35, hy=2.1, hz=0.35, cfg=cfg(mu=0.9), color=GREY)
        # Dry-stacked brick wall downrange (avbd-metal's castle is mortared; the port has no breakable joints).
        brick = (1.1, 0.55, 0.55)
        for h in range(8):
            # Running bond: offset courses carry one brick fewer so no brick overhangs the end.
            for i in range(10 - h % 2):
                y = -4.5 * brick[0] + i * brick[0] + (brick[0] / 2 if h % 2 else 0)
                body = b.add_body(xform=tf((31.5, y, h * brick[2] + brick[2] / 2)))
                b.add_shape_box(body, hx=brick[1] * 0.93 / 2, hy=brick[0] * 0.93 / 2, hz=brick[2] / 2, cfg=cfg(density=800, mu=0.85), color=[(0.72, 0.45, 0.33), (0.62, 0.38, 0.28)][(h + i) % 2])
        return b, {}, "Trebuchet: a hinged arm with a hung counterweight throws a boulder from its cup at a dry-stacked brick wall."



if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
