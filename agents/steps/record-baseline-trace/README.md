# Record a baseline trace

> **Does:** freezes a working PhysX project and records a deterministic per-step trace, keyed by body and joint names, to compare everything else against.
> **Use in:** port-from-physx.
> **Passes when:** the frozen project runs from its own launcher, and two recordings with the same seed agree within a repeatability envelope you declared before recording.

**Inputs.** The original project at a commit; its assets, checkpoints or scripted trajectories; its dependency
versions. **Tools.** The project's own environment and launcher, plus a recorder hooked into its step loop.

1. **Inventory.** Record commits of the application, every nested repository, assets, checkpoints and trajectory
   data, and hash the data files. Branch names are hints, not pins.
2. **Freeze.** Put the project in its own checkout and environment with its own installer and launcher. Keep
   `PYTHONPATH`, editable installs, generated assets and caches from crossing into any other tree; print the
   imported module roots at startup (child processes too) and check them.
3. **Record.** Fixed seeds; no domain randomization; no disturbances or observation noise. Per step, keyed by
   semantic body and joint names (never by index; backends order bodies differently), save: observations,
   commands, actions, joint targets, applied torques, body and joint state, contact and collision metadata,
   rewards and terminations. Include the reset state and the first observation and action.
4. **Repeat** the same recording and diff the two. Declare the envelope first (exact equality where the baseline
   is deterministic, otherwise per-quantity tolerances). Differences inside it are the baseline's own variability
   and set the floor for every later comparison; a difference outside it is a baseline problem to fix first.

**Outputs.** The frozen checkout and launcher, the pins and hashes, and the trace files (one per run, with the
command that made them).
