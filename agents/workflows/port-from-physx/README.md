# Port from PhysX

> **Goal:** a project (training environment, data collection, benchmark) that runs on PhysX and on Steel from one application behind one switch, with every remaining difference attributed to the solver.
> **Use when:** adding Steel to a codebase that runs on PhysX, or finding out why a ported task behaves differently under Steel.
> **Done when:** the upgraded-PhysX gate and the Steel parity check pass or every difference is explained, randomized and performance comparisons are matched and free of capacity loss, and a fresh checkout runs both backends.

**Inputs.** The original project at a known commit with its assets, checkpoints or scripted trajectories and
dependency versions. **Principles.** One change at a time (upgrade the framework with PhysX, then switch the
solver, then tune); pin everything; compare samples, not only aggregates; a clean exit is not a pass. Failure
signatures from earlier ports are in [lessons.md](lessons.md).

## Steps

1. [record-baseline-trace](../../steps/record-baseline-trace/README.md): freeze the original PhysX project and record
   its deterministic trace.
2. Upgrade the framework to a version that supports Newton, still on PhysX, then
   [check-parity](../../steps/check-parity/README.md) against the baseline.
   - Parity fails → fix the stack (reset timing, quaternion conventions, asset import, observation history). Do not
     go on: no Steel result can be interpreted until upgraded PhysX matches.
3. Add Steel behind one flag or preset in the same upgraded application. Construct `SolverSteel` with
   `row_watermark=True`. Smoke-test import, reset, zero-action stepping, controls, contacts, external wrenches and
   viewers before running a policy or planner. Treat a wrong-looking viewer as a visualization bug until the
   trajectory data says otherwise.
4. [check-parity](../../steps/check-parity/README.md), Steel against upgraded PhysX, no randomization.
   - Dropped rows → [size-capacities](../../steps/size-capacities/README.md), then repeat.
   - A difference you cannot explain → [debug-instability](../debug-instability/README.md) with the PhysX trace as
     the reference.
5. [randomize-one-factor](../../steps/randomize-one-factor/README.md) for each randomization factor, then the full
   randomization as its own comparison.
6. Training or collection smoke: state what a resumed checkpoint restores (from scratch, full state, actor only),
   verify each restored component and its shapes, and run one collection plus one update (or one collected episode).
   - It fails, or a deterministic rollout ends early → stop before any expensive run.
7. [benchmark-throughput](../../steps/benchmark-throughput/README.md), PhysX against Steel, at environment stepping
   without cameras and at the full application; tune Steel in the step's order with
   [tune-one-knob](../../steps/tune-one-knob/README.md), behavior gates fixed.
8. If policies are trained on each backend, evaluate both in a 2 x 2 policy-by-solver matrix and report completion
   and termination reasons next to reward per surviving step.
9. From a fresh checkout of the exact revision and its nested pins, run both backends, check imported module roots
   and data hashes, and require each run's own completion marker.

## Stop conditions

Upgraded PhysX still differs materially from the baseline; a deterministic rollout ends unexpectedly or a one-update
smoke fails; any run reports dropped rows, contact-buffer overflow or non-finite state, whatever its exit code; a
faster setting has not passed the reference's behavior and safety gates.

## Deliverables

Reproducible installers and launchers for the frozen baseline and the upgraded application; the pins; each step's
evidence (traces, parity tables, randomization rows, benchmark data with invalid points marked); every difference
with its cause and fix; the exact commands. Keep general library fixes separate from task-specific settings: a
setting that won on one task is evidence for that task, not a default.
