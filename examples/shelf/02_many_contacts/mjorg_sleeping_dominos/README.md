# Sleeping dominos (mujoco.org)

Steel replica of "Island Sleeping" from the MuJoCo 4.0 tour on mujoco.org. 96 dominoes stand in five rows
joined by U-turns. One tilted domino starts a wave; the dominoes ahead of it sleep until it reaches them,
and the fallen ones go back to sleep.

- Source: `google-deepmind/mujoco` `model/sleep/dominos.xml`, Apache-2.0 (text in `LICENSE`).
  `scene.xml` is that file compiled with MuJoCo 3.17 and re-emitted flat (its `replicate` blocks
  expanded), without the 4.0 `sleep` flag and `sleep_tolerance`. Steel's `enable_sleeping` replaces them.
- Check: after 15 s every domino leans more than 45 degrees from vertical. In Steel the wave reaches all
  96 by 8 s, as it does in MuJoCo 3.17 on the same file, and 96 of 96 are asleep by 9.5 s.

Known gaps:

- Steel's sleep thresholds (body speed 0.05 m/s, 0.15 rad/s for 0.5 s) differ from MuJoCo's
  `sleep_tolerance`.
- The shelf viewer does not mark sleeping bodies; the tour colours them.
- Rendering uses the shelf viewer, not MuJoCo's Filament renderer.
