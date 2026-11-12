# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Box turn
#
# Two arms hold a box between their fingertips only, flip it onto its side and then turn it a quarter turn.
#
# Rebuilt from steel-webgpu's exported scene (assets/scenes/box_turn: scene.json
# and scene.bin, copied next to this file); see common/exported_scene.py. The
# visual meshes draw when a steel-webgpu checkout sits next to the shelf.
#
# scene.json filters each arm's wrist_2 link against its own hand, which it
# overlaps at rest; the export filters only parent-child pairs.
#
# Command: python -m shelf run box_turn
#
###########################################################################

import numpy as np

import newton
import newton.examples
from common.exported_scene import ExportedSceneExample


class Example(ExportedSceneExample):
    file = __file__

    def build(self):
        builder, twists = super().build()
        # Each arm is a UR10e with a Franka hand bolted on. The export filters the UR10e's own link pairs and
        # the hand's, but not the pairs between them, and the wrist_2 capsule sits 21 mm inside the hand hull
        # at the start pose. Those contacts lock wrist_3 (it never follows its path) and drag the other joints
        # off theirs, so the fingertips miss the box. The arms do not touch themselves on this path: filter
        # every shape pair inside one arm, as the two halves already are.
        art_of = {child: builder.joint_articulation[j] for j, child in enumerate(builder.joint_child)}
        arm_shapes: dict[int, list[int]] = {}
        for shape, body in enumerate(builder.shape_body):
            art = art_of.get(body, -1) if body >= 0 else -1
            if art >= 0 and builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                arm_shapes.setdefault(art, []).append(shape)
        for shapes in arm_shapes.values():
            for i, a in enumerate(shapes):
                for b in shapes[i + 1 :]:
                    if builder.shape_body[a] != builder.shape_body[b]:
                        builder.add_shape_collision_filter_pair(a, b)
        return builder, twists

    def test_post_step(self):
        box = self.model.body_label.index("turn_box")
        self._box = box
        self._top = max(getattr(self, "_top", 0.0), float(self.state_0.body_q.numpy()[box, 2]))
        self._pinch = max(getattr(self, "_pinch", 0.0), self.deepest_contact(bodies=[box]))

    def check(self):
        """The arms lift the box (above 0.25 m), flip it onto its side and turn it a quarter turn: its x axis ends
        along world y, y along z and z along x within 5 deg, set down within 3 cm of where it started. The fingertips
        pinch the box less than 5 mm deep (about 3 mm here; 4 x 8 reaches 5.5 mm and 2 x 16 7.4 mm, and even 1 x 4
        still turns the box, so the pinch is what the settings buy)."""
        q = self.state_0.body_q.numpy()[self._box]
        start = self.model.body_q.numpy()[self._box]
        x, y, z, w = q[3:]
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])  # fmt: skip
        goal = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # columns: where the box's x, y, z axes should point
        worst = np.degrees(np.arccos(np.clip(np.sum(rot * goal, axis=0), -1, 1))).max()
        assert self._pinch < 0.005, f"the fingertips pinch the box {1e3 * self._pinch:.1f} mm deep"
        assert self._top > 0.25, f"the box only rose to {self._top:.3f} m"
        assert worst < 5.0, f"the box ends {worst:.1f} deg off its goal orientation"
        assert np.linalg.norm(q[:2] - start[:2]) < 0.03 and q[2] < 0.09, (
            f"the box ends at {np.round(q[:3], 3)}, not set down"
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
