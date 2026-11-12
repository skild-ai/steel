# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI
# SPDX-License-Identifier: Apache-2.0
# Direct source: steel-webgpu; see presets.toml and THIRD_PARTY.md.
# Modified for Steel Shelf: preset-driven execution and shared runner integration.

###########################################################################
# Example Drawer and fork
#
# The arm pulls an unpowered drawer open by its handle, then picks up a fork and places it in the drawer's tray.
#
# Rebuilt from steel-webgpu's exported scene (assets/scenes/drawer_fork: scene.json
# and scene.bin, copied next to this file); see common/exported_scene.py. The
# visual meshes draw when a steel-webgpu checkout sits next to the shelf.
#
# Command: python -m shelf run drawer_fork
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
