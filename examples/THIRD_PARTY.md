# Third-party licenses and attribution

Reviewed 2026-11-09; migration reconciled with Steel Shelf main on 2026-11-11. This file declares the terms of the tracked code, scene
exports, documentation artwork and the asset references used by Steel Shelf.
The exact files, source revisions, notices and content hashes are recorded in
[third_party_inventory.json](third_party_inventory.json).

## What the package contains

Project source code and original project documentation artwork are Apache-2.0.
The package also contains exported robot model data under BSD-3-Clause, so the
wheel and sdist declare **`Apache-2.0 AND BSD-3-Clause`**. These are the terms of
distinct components, not a choice of license for every file.

Both distributions include all 13 tracked `scene.json` / `scene.bin` pairs and
the full notices below. The same model terms apply to both parts of each
export. They do not vendor optional external visual-mesh files, URDF/USD models,
policies, installed dependencies, or downloaded asset caches.
Documentation images remain in the Git repository and are omitted from both
Python distribution archives because they are not runtime data.

## Bundled scene data

The exports came through steel-webgpu. Procedural tabletop/scene content is
Apache-2.0; the robot-derived content retains these additional notices:

| Exported content | Scenes | Required notice |
| --- | --- | --- |
| Universal Robots UR10e | cereal, spinning_top, brick_pour, box_turn, toaster, toy_train | [BSD-3-Clause, Copyright 2018 ROS Industrial Consortium](licenses/UR10e-BSD-3-Clause.txt) |
| Universal Robots UR5e | air_hockey | [BSD-3-Clause, Copyright 2018 ROS Industrial Consortium](licenses/UR5e-BSD-3-Clause.txt) |
| Kinova Gen3 | viaduct_kit | [BSD-3-Clause, Copyright 2018 Kinova inc.](licenses/Kinova-Gen3-BSD-3-Clause.txt) |
| Franka model and gripper content | beer_pong, jenga, balance_board, drawer_fork; gripper components in other exports where used | [Complete upstream license: Apache-2.0 plus appended BSD/Willow Garage terms](licenses/Franka-complete-LICENSE.txt) |
| Procedural scene without a robot | lava_fall | [Apache-2.0](LICENSE) |

The UR10e, UR5e and Gen3 identifications were checked against the scene metadata and
all model link masses in the pinned Menagerie models. The matching
source URLs and values are recorded in the inventory. The Franka notice is
preserved in full, including the BSD section after its Apache text. Generated
bodies, joints, inertias and collision geometry are model data even when the
original mesh or URDF is absent.

## Source-code attribution

Newton-derived examples retain their original copyright and Apache-2.0 SPDX
headers. The pinned Newton dependency is
[f6fc82a8](https://github.com/dylanturpin/newton-collab/blob/f6fc82a8b06d036e1e6d02a61d77a2d300476e8d/LICENSE.md).

The 33 Steel WebGPU ports (20 builder scenes and 13 exported scenes) identify
**steel-webgpu as their direct source** in the source header and `presets.toml`.
The [steel-webgpu LICENSE.md](https://github.com/skild-ai/steel-webgpu/blob/f2f72c1e99aea7192cf2366eb918e630135afcf9/LICENSE.md)
provides Apache-2.0 code terms and, under *Bundled models*, the UR10e, UR5e and
Kinova Gen3 notices used above. It lists the Franka model under Apache-2.0; the
appended BSD/Willow Garage terms are supplemented above.

The 13 `scene.json` / `scene.bin` pairs can be checked against public
[steel-webgpu](https://github.com/skild-ai/steel-webgpu) `assets/scenes/`:

- All 13 `scene.bin` files are byte-identical to
  [`f2f72c1`](https://github.com/skild-ai/steel-webgpu/tree/f2f72c1e99aea7192cf2366eb918e630135afcf9/assets/scenes)
  (unchanged since the first public commit).
- Eight `scene.json` files are byte-identical to
  [`58246f3`](https://github.com/skild-ai/steel-webgpu/tree/58246f3d3c6647edc66dd8518e50f13641418c38/assets/scenes):
  air_hockey, balance_board, beer_pong, brick_pour, drawer_fork, spinning_top,
  toy_train and viaduct_kit. The `box_turn` and `toaster` files matched it
  before a collision-filter-only edit (see below).
- The `cereal`, `jenga` and `lava_fall` JSON were exported from steel-webgpu at
  a pre-release revision that is not in its public history. They differ from
  the public files in solver settings, initial state and a few friction values;
  their binaries match.

The 20 builder scenes come from `tools/scenes.py`, whose functions are in
public steel-webgpu from `58246f3` on. The `@ ad03e49` in their `presets.toml`
`source` lines names the same pre-release revision. The inventory records the
public commit and hash for every file, and the audit checks all 26 current files
and both distributions against those hashes.

Existing copyright headers are preserved. “Modified for Steel Shelf” marks
only the later adaptation; no copyright holder has been reassigned by this review.

## Models and policies downloaded at runtime

These assets are not bundled in the repository, wheel or sdist. Copies of
their full license notices are supplied here for attribution and for anyone
assembling an asset cache. Keep the applicable notices with redistributed
models or policies; the root code license does not replace them.

The default newton-assets revision is
`f8fb7abcbeba2318814a74f3eeb02780ad7925d6`. The DR Legs example overrides it with
`261cd1f429619d8ef4f546bd788ab9dea906b5e1`. Menagerie and IsaacGymEnvs references
are pinned below and in the inventory.

| Asset | Terms | Evidence |
| --- | --- | --- |
| `anybotics_anymal_c/rl_policies` | Apache-2.0 | [Full notice](licenses/runtime-assets/anybotics_anymal_c--rl_policies.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/anybotics_anymal_c/rl_policies/LICENSE) |
| `anybotics_anymal_c` | BSD-3-Clause | [Full notice](licenses/runtime-assets/anybotics_anymal_c.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/anybotics_anymal_c/LICENSE) |
| `anybotics_anymal_d` | BSD-3-Clause | [Full notice](licenses/runtime-assets/anybotics_anymal_d.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/anybotics_anymal_d/LICENSE) |
| `asroballet/rl_policies` | Apache-2.0 | [Full notice](licenses/runtime-assets/asroballet--rl_policies.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/asroballet/rl_policies/LICENSE) |
| `asroballet` | MIT | [Full notice](licenses/runtime-assets/asroballet.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/asroballet/LICENSE) |
| `disneyresearch/dr_legs` | LicenseRef-Disney-Research | [Full notice](licenses/runtime-assets/disneyresearch--dr_legs.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/261cd1f429619d8ef4f546bd788ab9dea906b5e1/disneyresearch/dr_legs/LICENSE) |
| `disneyresearch/dr_testmech` | LicenseRef-Disney-Research | [Full notice](licenses/runtime-assets/disneyresearch--dr_testmech.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/disneyresearch/dr_testmech/LICENSE) |
| `franka_emika_panda` | Apache-2.0 AND BSD-3-Clause | [Full notice](licenses/runtime-assets/franka_emika_panda.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/franka_emika_panda/LICENSE) |
| `manipulation_objects/cup` | Apache-2.0 | [Full notice](licenses/runtime-assets/manipulation_objects--cup.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/manipulation_objects/cup/LICENSE) |
| `manipulation_objects/pad` | Apache-2.0 | [Full notice](licenses/runtime-assets/manipulation_objects--pad.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/manipulation_objects/pad/LICENSE) |
| `manipulation_objects/rubber_duck` | Apache-2.0 | [Full notice](licenses/runtime-assets/manipulation_objects--rubber_duck.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/manipulation_objects/rubber_duck/LICENSE) |
| `style3d` | BSD-3-Clause | [Full notice](licenses/runtime-assets/style3d.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/style3d/LICENSE) |
| `unitree_g1/rl_policies` | Apache-2.0 | [Full notice](licenses/runtime-assets/unitree_g1--rl_policies.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/unitree_g1/rl_policies/LICENSE) |
| `unitree_g1` | BSD-3-Clause | [Full notice](licenses/runtime-assets/unitree_g1.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/unitree_g1/LICENSE) |
| `unitree_go2/rl_policies` | Apache-2.0 | [Full notice](licenses/runtime-assets/unitree_go2--rl_policies.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/unitree_go2/rl_policies/LICENSE) |
| `unitree_go2` | BSD-3-Clause | [Full notice](licenses/runtime-assets/unitree_go2.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/unitree_go2/LICENSE) |
| `unitree_h1` | BSD-3-Clause | [Full notice](licenses/runtime-assets/unitree_h1.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/unitree_h1/LICENSE) |
| `universal_robots_ur10` | BSD-3-Clause | [Full notice](licenses/runtime-assets/universal_robots_ur10.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/universal_robots_ur10/LICENSE) |
| `wonik_allegro` | BSD-2-Clause | [Full notice](licenses/runtime-assets/wonik_allegro.txt) · [Pinned source](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/wonik_allegro/LICENSE) |
| `MuJoCo Menagerie UR10e` | BSD-3-Clause | [Full notice](licenses/UR10e-BSD-3-Clause.txt) · [Pinned source](https://github.com/google-deepmind/mujoco_menagerie/blob/f054586a8e90465d49ee5be15335c4a0c7f57caf/universal_robots_ur10e/LICENSE) |
| `IsaacGymEnvs factory nut/bolt` | BSD-3-Clause | [Full notice](licenses/runtime-assets/IsaacGymEnvs-BSD-3-Clause.txt) · [Pinned source](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/LICENSE.txt) |

### Disney Research terms

`disneyresearch/dr_testmech` and `disneyresearch/dr_legs` permit use only for
research or software development, including benchmarking. Redistributions must
retain the copyright, conditions and disclaimer. Their names must not imply
endorsement, sponsorship or affiliation without permission. These are custom
field-of-use terms, not Apache-2.0 or an unrestricted open-source model license.
The two examples download the models separately; neither model is shipped here.
The complete terms are included in the linked notice files above.

### IsaacGymEnvs factory meshes

The two nut/bolt examples now pin their download to
`aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3`.
The [factory acknowledgments](licenses/runtime-assets/IsaacGymEnvs-factory-acknowledgments.txt)
identify the nut and bolt as custom-designed assets distributed under the
project's BSD-3-Clause terms. This declaration covers the nut/bolt folder used
by these examples, not unrelated IsaacGymEnvs assets.

## Newton dependency assets

`newton.examples.get_asset()` resolves files inside the pinned Newton
installation, not inside this project's packages. The dependency distributes
its built-in example assets under its
[Apache-2.0 repository license](https://github.com/dylanturpin/newton-collab/blob/f6fc82a8b06d036e1e6d02a61d77a2d300476e8d/LICENSE.md).
All 31 files in that asset directory are enumerated by path and Git blob ID
in the inventory, and the Shelf call sites are listed separately. This includes
bunny, bear, Crazyflie, RJ45, cloth, cartpole, quadruped, ant/humanoid and the
procedural tabletop assets. These dependency files are not copied into the
Steel Shelf wheel or sdist.

The tabletop source notes identify the crane and mug as procedurally generated.
The [Pelican bicycle note](https://github.com/dylanturpin/newton-collab/blob/f6fc82a8b06d036e1e6d02a61d77a2d300476e8d/newton/examples/assets/steel_tabletop/pelican_bicycle.md)
credits original CAD by liinbor under Apache-2.0 and links its permanent source
tag. The Newton helpers separately load Franka and Menagerie robot models;
those model notices are preserved above rather than inferred from Newton's code license.

### Newton test and Warp example assets

The two optional Kamino USD-loading examples use
`newton.tests.get_kamino_basics_asset()`, which resolves six basic test scenes
inside the pinned Newton dependency. Their paths and Git blob IDs are recorded
separately in the inventory. They are covered by Newton's Apache-2.0 repository
license; the files identify manually authored test scenes, with the cartpole
file crediting Disney Research as its generator. These files are separate
from the custom-licensed Disney robot downloads described above.

The `cloth_twist` and `cloth_stiff_material_hanging` examples use
`warp/examples/assets/square_cloth.usd` from the separately installed Warp
package. The locked Warp 1.18.0 source distributes this asset under its
[Apache-2.0 repository license](https://github.com/NVIDIA/warp/blob/f2eaed82d8d03b37bf1014cc975954067fefcf16/LICENSE.md);
its separate-license directory lists no square-cloth exception. The pinned
source blob and license hash are recorded in the inventory. Other installed
Warp versions retain their supplied notices. None of these dependency assets
is copied into Steel Shelf's wheel or sdist.

## Documentation artwork

These are project-created charts and rendered example recordings, not imported
stock photos or third-party media. Original Steel Shelf chart/render artwork
is released under Apache-2.0. Applicable model notices above are retained for
the depicted model content; this does not relicense the underlying models.
The media's SHA-256 hashes are in the inventory.

| File | Creation |
| --- | --- |
| `docs/steel_tuning/frame_time_dark.png` | docs/steel_tuning/charts.py with evidence.json |
| `docs/steel_tuning/frame_time_light.png` | docs/steel_tuning/charts.py with evidence.json |
| `docs/steel_tuning/summary_dark.png` | docs/steel_tuning/charts.py with evidence.json |
| `docs/steel_tuning/summary_light.png` | docs/steel_tuning/charts.py with evidence.json |
| `docs/teaser.webp` | Recorded from the shelf's own examples |
| `docs/tuner.gif` | Recorded from the shelf's own tuner |

## Optional external files and Python dependencies

`STEEL_WEBGPU_ASSETS` can supply visual meshes from a user's separate checkout.
Those files are not distributed here. The checked WebGPU model notices are
listed above; preserve the supplied source's notices when assembling that
external asset directory. User-supplied replacement files retain their own terms.

Installed dependencies remain separate distributions with their own license
files. For example, certifi is MPL-2.0 and pyglet is BSD-3-Clause. Steel Shelf
does not vendor either dependency; its package license expression describes
its own shipped files, not a relicensing of the Python environment.

## Distribution checks

`python scripts/check_distribution_licenses.py DIST_DIR` checks the wheel and
sdist: exact JSON and binary scene bytes, inventory coverage, each complete notice and its
License-File metadata, robot-model notices, and absence of external assets and
documentation images. It also rejects the original Apache-only wheel and a
wheel with the BSD notice removed. This declaration review does not change
solver behavior or grant rights beyond the stated component licenses.

Runtime call sites are compared by function, arguments and keyword values,
not line numbers or traversal order. After reviewing an intentional asset
reference change, refresh those records with
`python scripts/check_distribution_licenses.py --update-call-sites`.
Line numbers are retained only to help navigate the source.

These checks enforce the recorded inventory and archive contents; they do not
resolve arbitrary runtime Python expressions, establish original authorship,
or automatically approve a newly added asset-loading mechanism. New helpers,
changed dependency revisions and dynamic asset selections still need source
review, even when the archive check passes.

## Additions already present in the migrated main branch

The `box_turn` and `toaster` JSON files add collision filter pairs to the
exported scenes (toaster's carriage against its bread-guide wires, box_turn's
wrist against its own hand); their binaries and robot model data are unchanged.
The inventory retains both the original JSON hash and adapted hash.

The seven `mjorg_*` examples retain MuJoCo XML/model assets under Apache-2.0,
including the card mesh and textures for the house of cards. Their per-example
README files describe the source and transformations, and their LICENSE files are
included in both distributions. Original linkage reference JSON/NPZ trajectories
are project-generated Apache-2.0 data; their generating scripts are retained.
`additional_bundled_assets` lists the exact files, hashes, notices and provenance.
