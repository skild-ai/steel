# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Linkages
#
# Mimic joints and a revolute loop closure: mirrored pendulums and a crank-driven four-bar.
#
# Ported from steel-webgpu's tools/scenes.py (linkages); the scene builds
# as it does there and the shelf's preset drives the solver.
#
# Command: python -m shelf run linkages
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
    camera = {'eye': (0.6, -1.5, 0.8), 'target': (0.05, 0.0, 0.4)}

    def build(self):
        """Mimic joints and a revolute loop closure: mirrored pendulums and a crank-driven four-bar."""
        b = newton.ModelBuilder()
        self.preset.apply_model(b)  # the preset's builder settings (the contact gap), before any shape
        b.add_ground_plane()
        bar = cfg(density=600)
        # Mimic: the leader swings freely; the followers copy it with multipliers -1 and 0.5.
        pend = []
        for k, (y, color) in enumerate(((-0.15, GOLD), (0.0, BLUE), (0.15, TEAL))):
            link = b.add_link(xform=tf((-0.3, y, 0.45)))
            b.add_shape_capsule(link, radius=0.015, half_height=0.11, cfg=bar, color=color)
            b.add_shape_sphere(link, xform=tf((0, 0, -0.13)), radius=0.035, cfg=bar, color=color)
            j = b.add_joint_revolute(-1, link, parent_xform=tf((-0.3, y, 0.6)), child_xform=tf((0, 0, 0.15)), axis=(0, 1, 0))
            pend.append(j)
        b.add_articulation(pend)
        b.set_joint_mimic(pend[1], pend[0], coeffs=(0.0, -1.0))
        b.set_joint_mimic(pend[2], pend[0], coeffs=(0.0, 0.5))
        # Four-bar: ground pivots A and D, crank AB driven at 2 rad/s, rocker closed to D by a revolute loop.
        A = np.array((0.15, 0.0, 0.3)); D = np.array((0.45, 0.0, 0.3))
        B = A + (0.0, 0.0, 0.1)
        lc, lr = 0.3, 0.2
        d = np.linalg.norm(D - B); a = (lc**2 - lr**2 + d**2) / (2 * d); h = math.sqrt(lc**2 - a**2)
        u = (D - B) / d; perp = np.array((-u[2], 0.0, u[0]))
        C = B + a * u + h * perp
        def seg(body, p0, p1, color):
            mid, v = (np.asarray(p0) + np.asarray(p1)) / 2, np.asarray(p1) - np.asarray(p0)
            q = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*(v / np.linalg.norm(v))))
            b.add_shape_capsule(body, xform=tf(mid - p0, q), radius=0.012, half_height=float(np.linalg.norm(v)) / 2, cfg=bar, color=color)
        crank = b.add_link(xform=tf(A)); seg(crank, A, B, RED)
        coupler = b.add_link(xform=tf(B)); seg(coupler, B, C, GOLD)
        rocker = b.add_link(xform=tf(C)); seg(rocker, C, D, BLUE)
        y = (0, 1, 0)
        j0 = b.add_joint_revolute(-1, crank, parent_xform=tf(A), axis=y, target_vel=2.0, target_kd=5.0)
        j1 = b.add_joint_revolute(crank, coupler, parent_xform=tf(B - A), axis=y)
        j2 = b.add_joint_revolute(coupler, rocker, parent_xform=tf(C - B), axis=y)
        b.add_articulation([j0, j1, j2])
        b.add_joint_revolute(-1, rocker, parent_xform=tf(D), child_xform=tf(D - C), axis=y)
        start = {"leader": (pend[0], (0.8,)), "mirror": (pend[1], (-0.8,)), "half": (pend[2], (0.4,))}
        return b, {}, "Mimic and loop closure: two pendulums copy the gold one (x -1, x 0.5); a crank drives a four-bar closed by a revolute loop.", start

    def test_post_step(self):
        q = self.state_0.joint_q.numpy()
        self._mimic_err = max(getattr(self, "_mimic_err", 0.0), abs(q[1] + q[0]), abs(q[2] - 0.5 * q[0]))
        self._frames = getattr(self, "_frames", 0) + 1
        if self._frames > 200:  # the leader's swing after more than three periods
            self._amp = max(getattr(self, "_amp", 0.0), abs(q[0]))
        self._crank0 = getattr(self, "_crank0", q[3])

    def check(self):
        """The copies hold -1 and 0.5 of the leader, the leader keeps its 0.8 rad swing, the crank turns at 2 rad/s."""
        q = self.state_0.joint_q.numpy()
        assert self._mimic_err < 1e-3, f"a mimic copy is {self._mimic_err:.4f} rad off"
        assert self._amp > 0.75, f"the leader's swing decayed to {self._amp:.3f} rad"
        rate = (q[3] - self._crank0) / ((self._frames - 1) * self.frame_dt)
        assert abs(rate - 2.0) < 0.05, f"the crank turns at {rate:.3f} rad/s"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
