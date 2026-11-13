# Randomize one factor

> **Does:** turns on domain randomization one factor at a time and compares the backends under each.
> **Use in:** port-from-physx.
> **Passes when:** each factor's samples and deltas stay within the tolerance set at nominal parity, or the factor's difference has a named cause.

**Inputs.** Nominal parity passed (check-parity); the task's randomization factors (masses, friction, restitution,
gains, delays, pushes, initial states). **Tools.** The application's randomization config and backend switch.

1. With nominal settings matching, enable one factor; keep every other factor at its nominal value and the seeds
   fixed.
2. Run both backends and compare per-sample values and deltas, not only mean return: the sampled values each solver
   actually reads, the trajectories, terminations and contact behavior.
3. When a factor differs, ablate inside it (for materials: friction alone, restitution alone) until one property
   explains it. Check that the property reaches the solver after randomization and that a solver option for it is
   not silently dropped.
4. Only after every factor passes alone, run the full randomization as its own labeled comparison.

**Rules.** A choice that fixes one task (for example restitution off) can break another; make task-specific
choices explicit in that task's launcher, not a shared default.

**Outputs.** One row per factor: samples compared, deltas, cause if any, fix or acceptance.
