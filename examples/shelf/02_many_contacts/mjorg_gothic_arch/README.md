# Gothic arch (mujoco.org)

Steel replica of "Dry-Stone Gothic Arch" from the MuJoCo 4.0 tour on mujoco.org. Sixteen voussoirs and a
keystone sit between two fixed springers with no mortar and stand on friction alone.

- Source: `google-deepmind/mujoco` `model/arch/gothic.xml`, Apache-2.0 (text in `LICENSE`). `scene.xml` is that file
  compiled with MuJoCo 3.17 and re-emitted flat (its `replicate` blocks expanded, the 4.0 discrete
  integrator replaced by `implicitfast`).
- Check: after 10 s the keystone has sunk less than 10 mm, no stone has moved more than 25 mm, and no
  stone has moved more than 5 mm over the last 5 s. MuJoCo 3.17 on the same file sinks the keystone 1 mm
  and moves no stone more than 2 mm.
  The total bounds are loose on purpose: the stones stick and slip in the first seconds (below), and
  bounds of 5 mm and 10 mm failed about one run in eight although the arch then held.

Known gaps:

- Steel needs 0.5 ms steps (32 substeps at 60 fps) where MuJoCo takes 8 ms with an elliptic cone,
  `impratio` 10 and two noslip iterations; Steel has no noslip pass.
- The stones stick and slip: in some runs the lowest voussoirs slip outward by up to about 20 mm in the
  first seconds, then the arch locks. Over 12 runs on an RTX PRO 6000 the keystone sank 0.6-5.7 mm in 10 s
  and at most 4.6 mm in 20 s.
- The arch never comes fully to rest in Steel: over 40 s every setting tried (up to 32 x 96 and 64 x 48
  substeps x iterations) creeps, the keystone sinking about 1 mm and the stones moving 2-3 mm per 20 s; the
  preset's 32 x 20 creeps at that rate.
- Rendering uses the shelf viewer, not MuJoCo's Filament renderer.
