# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Spinning top
#
# A motorised chuck grips the top's stem, spins it up and releases it; the top keeps spinning upright on its tip.
#
# Rebuilt from steel-webgpu's exported scene (assets/scenes/spinning_top: scene.json
# and scene.bin, copied next to this file); see common/exported_scene.py. The
# visual meshes draw when a steel-webgpu checkout sits next to the shelf.
#
# Command: python -m shelf run spinning_top
#
###########################################################################

import newton.examples
from common.exported_scene import ExportedSceneExample


class Example(ExportedSceneExample):
    file = __file__


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
