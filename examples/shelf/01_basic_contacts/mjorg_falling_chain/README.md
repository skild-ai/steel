# Falling chain (mujoco.org)

Steel replica of "Speculative Contacts: Falling Chain" from the MuJoCo 4.0 tour on mujoco.org. Four
interlocked chain links built from 6 mm capsules drop 8 cm onto a catch bar at a coarse time step. With
speculative contacts the links stay interlocked and the chain hangs from the bar.

- Source: `google-deepmind/mujoco` `model/speculative/chain.xml`, Apache-2.0 (text in `LICENSE`). `scene.xml` is that file
  compiled with MuJoCo 3.17 and re-emitted flat, without the 4.0 `speculative` flag (Newton's collision
  pipeline enables speculative contacts instead, from the preset's `collision` table).
- Check: after 10 s link 1 hangs on the bar and each link hangs 25-34 cm below the one above (they start
  18 cm apart). Without speculative contacts links 3 and 4 fall through, as they do in MuJoCo 3.17.

Known gaps:

- Newton sizes the speculative gap from each body's velocity at the start of the step, capped by
  `speculative_contact_gap_max`. MuJoCo 4.0's speculative contacts are its own scheme.
- The step is 6.7 ms (3 substeps at 50 fps); the tour uses 6 ms.
- Rendering uses the shelf viewer, not MuJoCo's Filament renderer, image-based light or skybox.
