# Failure signatures from earlier ports

Part of [port-from-physx](README.md). Each item is something that made a ported task behave
differently, and the check that found it. They come from ports of locomotion tracking, multi-agent rigid-object,
detailed-mesh manipulation and planner-driven manipulation tasks from PhysX to a Newton solver; the solver-specific
details differ for Steel, the checks do not. Treat them as things to check, not as defaults.

## Before the solver changes (upgraded-PhysX gate)

- **Reset timing, quaternion conventions and asset import** changed between framework versions. The upgraded PhysX
  build failed parity until each was fixed; none of them was a solver difference.
- **Stale observation history** after a reset differed between backends on the first step only. Fix or disclose
  it before reading closed-loop error as a solver delta.
- **Data shipped in separate archives** (motions and labels), or with absolute symlinks that broke after
  extraction. Stage and hash every payload.
- **Two checkouts imported each other's code** through `PYTHONPATH` or editable installs. Print and check the
  imported module roots in every run, including spawned child processes, which do not inherit a parent's
  `sys.path` edits.

## Actuation and forces

- **Explicit actuator gains applied twice**, once by the framework and once by the backend adapter.
- **External wrenches cleared after the first internal substep.** Forces and targets applied by the application
  must hold for every solver substep.
- **A solver option dropped by a keyword filter.** The framework declared restitution off, the backend adapter did
  not pass the flag, and nominal parity still looked good. Randomized material runs regressed. Check that every
  option reaches the solver, and ablate material properties one at a time.
- **A blanket default that helped one task hurt others.** Turning restitution off recovered a tracking task and
  degraded ball and object-contact tasks. Make task-specific choices explicit in that task's launcher.

## Ordering and selection

- **Body and joint traversal order differs between backends.** Policy observation and action selectors, reward
  body lists and trace comparisons must use explicit semantic name order.

## Contacts and collision model

- **Collision APIs on instanced ancestors were not inherited** by the leaf meshes in the new backend; a collider
  overlay was needed. Check the finalized model's collision shapes, not the source asset's annotations.
- **A clone or import path substituted a convex hull** for a detailed mesh or SDF. Check the realized collision
  representation after every import path.
- **A fast free body needed its predictive contact horizon.** Shortening it globally brought back impact
  penetration; narrowing admission only for articulated pairs kept the free-body behavior.
- **A friction gap applied globally changed a ball's behavior** while aggregate reward looked fine. Scope such
  settings to the pairs they are meant for.
- **Higher contact counts were real but not an overflow.** Count contacts and rows separately, and read the
  overflow counters, before calling more contacts a capacity problem.
- **Independently randomized materials per collider** defeated contact reduction and made a reduction feature
  slower. Benchmark with the real randomization.

## Capacity

- **Collision candidates, retained contacts and each row family are separate capacities.** Raising one never fixes
  another.
- **The complete task horizon set the high-water mark**, not a start-up or short settle. Size capacities from full
  episodes over several seeds.
- **The busiest world's demand at a large world count was not predicted** by a smaller run, and varied between
  processes. Repeat target-scale screens.
- **The fastest settings at the largest world count all dropped rows.** Reject every point with candidate, contact
  or row loss whatever the exit status. If removing a whole row family still leaves an overflowing tail, stop tuning
  thresholds: the limit is structural.
- **A run of the reference backend exited 0 with a plausible throughput while overflowing its own contact buffer.**
  Log validation applies to both backends.

## Determinism and repeats

- **Deterministic contact ordering and sticky contact matching did not make full rollouts bitwise repeatable.**
  Outcome counts moved by a few items between identical processes, in both directions. Repeat before attributing a
  difference to a version.
- **A threshold miss is not a failure of the task.** Separate, for example, a lift height just under a threshold
  from an object actually dropped, using traces or video of the collision shapes.

## Viewers

- **Viewer transforms and cameras** needed explicit quaternion order conversion, a pumped update loop, framing once
  after reset (continuous tracking blocked manual camera control), and one pose publication per displayed frame
  rather than per physics substep.

## Tests and harnesses

- **A framework's test configuration changed a library setting** (for example disabling gradients) and produced a
  false failure in the solver's own tests. Run the solver's tests with its own runner and configuration.
- **A process can exit 0 with failures in its structured results, or pass every assertion and crash in teardown.**
  Require both valid results and a clean exit; keep teardown failures.
- **A saved "success" flag was not a physical gate.** An absolute height threshold passed an object that started
  above it; check relative change over the phase.
- **Destructor order and reference cycles** can make teardown crash after a run that otherwise passed. Capture
  unraisable exceptions in a subprocess and check terminal logs.
