# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Toaster
#
# The arm turns a bread slice upright, drops it into the second slot and presses the lever; the spring-loaded carriage latches down.
#
# Rebuilt from steel-webgpu's exported scene (assets/scenes/toaster: scene.json
# and scene.bin, copied next to this file); see common/exported_scene.py. The
# visual meshes draw when a steel-webgpu checkout sits next to the shelf.
#
# scene.json filters the carriage against the bread-guide wires it overlaps at
# its upper limit, so the prismatic joint alone governs the lift.
#
# Command: python -m shelf run toaster
#
###########################################################################

import numpy as np

import newton
import newton.examples
from common.exported_scene import ExportedSceneExample

CARRIAGE = "procgen/toaster/bread_lift_0"


class Example(ExportedSceneExample):
    file = __file__

    def build(self):
        builder, twists = super().build()
        # The bread guides are eight static wire capsules (r = 1 mm) leaning 1.8 degrees, and the carriage's slice
        # rests are 0.4 mm wider than the gap between them, so they overlap the wires by up to 1.3 mm wherever the
        # carriage is. The leaning contact normals then wedge the carriage up 4 cm past its upper limit in the first
        # frame, and the lever press jams it against the narrowing guides instead of latching it down. In a toaster
        # the carriage passes through the guides; only the bread rides on them, so the carriage ignores the wires.
        lift = builder.body_label.index(CARRIAGE)
        wires = [
            s
            for s in range(builder.shape_count)
            if builder.shape_body[s] == -1 and builder.shape_type[s] == newton.GeoType.CAPSULE
        ]
        carriage = [s for s in range(builder.shape_count) if builder.shape_body[s] == lift]
        for w in wires:
            for c in carriage:
                builder.add_shape_collision_filter_pair(w, c)
        # The toaster's shell and feet are welded to the world and sit 1.7 mm into the table: 32 contact rows every
        # step (two thirds of the scene's contacts) between bodies that cannot move. Skip them.
        welded = set()
        grew = True
        while grew:
            grew = False
            for j, kind in enumerate(builder.joint_type):
                child, parent = builder.joint_child[j], builder.joint_parent[j]
                if kind == newton.JointType.FIXED and child not in welded and (parent == -1 or parent in welded):
                    welded.add(child)
                    grew = True
        static = [s for s in range(builder.shape_count) if builder.shape_body[s] == -1]
        fixed = [s for s in range(builder.shape_count) if builder.shape_body[s] in welded]
        for s in static:
            for w in fixed:
                builder.add_shape_collision_filter_pair(s, w)
        return builder, twists

    def check(self):
        """The slice goes into the second slot and the press latches the carriage down: the carriage ends below -0.08
        (latched at -0.088) and the inserted slice sits in slot 2 beside the preloaded one, lowered with it."""
        q = self.state_0.joint_q.numpy()
        carriage = float(q[self.model.joint_q_start.numpy()[16]])  # bread_lift_0.q
        assert carriage < -0.08, f"the carriage is at {carriage:.4f}, not latched down"
        labels = self.model.body_label
        inserted = self.state_0.body_q.numpy()[labels.index("toaster/inserted-bread"), :3]
        preloaded = self.state_0.body_q.numpy()[labels.index("toaster/preloaded-bread"), :3]
        assert abs(inserted[2] - preloaded[2]) < 0.015, (
            f"the inserted slice is at z={inserted[2]:.3f}, the preloaded one at {preloaded[2]:.3f}"
        )
        assert abs(inserted[0] - 0.285) < 0.015 and abs(inserted[1] - 0.12) < 0.02, (
            f"the inserted slice ended at {np.round(inserted, 3)}, not in slot 2"
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
