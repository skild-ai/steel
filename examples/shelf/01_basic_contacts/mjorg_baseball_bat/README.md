# Baseball bat (mujoco.org)

Steel replica of "Speculative Contacts: Baseball Bat" from the MuJoCo 4.0 tour on mujoco.org. A spring
plunger pitches a 4 cm ball, and a spring-loaded bat 1.2 cm thick whips round and hits it, at a 10 ms
time step.

- Source: `google-deepmind/mujoco` `model/speculative/flipper.xml`, Apache-2.0 (text in `LICENSE`). `scene.xml` is that file
  compiled with MuJoCo 3.17 and re-emitted flat, without the 4.0 `speculative` flag (Newton's collision
  pipeline enables speculative contacts instead, from the preset's `collision` table).
- Check: the ball is pitched at more than 3 m/s towards the bat and then leaves with positive x velocity.
  Without speculative contacts it passes through the bat, as it does in MuJoCo 3.17.

Known gaps:

- Newton sizes the speculative gap from each body's velocity at the start of the step, capped by
  `speculative_contact_gap_max`. MuJoCo 4.0's speculative contacts are its own scheme.
- The pitch is a little faster than MuJoCo 3.17's on the same file (5.8 against 5.4 m/s).
- The bat motor is unused, as on the site. Rendering uses the shelf viewer, not MuJoCo's Filament
  renderer, and plays in sim time rather than the tour's 0.2x slow motion.
