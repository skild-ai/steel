# Debug instability

> **Goal:** the cause of a Steel scene's misbehavior (explosion, jitter, sinking, tunnelling, slipping grasp, open loop, overflow, NaN), and the smallest change that removes it.
> **Use when:** a scene misbehaves under Steel, a trial or autotune ends unhealthy, a render grades unstable, or a ported task diverges from PhysX.
> **Done when:** the cause is named with evidence, the fix passes the scene's gates over the whole task, or the report says why no setting can fix it.

**Inputs.** The scene and the settings that misbehave; when it goes wrong (time, frame, load); a trial or a render
of it if you have one.

## Steps

1. [run-trial](../../steps/run-trial/README.md) at the failing settings. Read the advice in order: the first fix is
   the one to make first; later numbers are unreliable until it is made.
2. Decide by the first problem:

   | First problem | Next |
   |---|---|
   | dropped rows (`*_capacity`), contact buffer full | [size-capacities](../../steps/size-capacities/README.md), then step 1 again |
   | start state written to `body_q`/`body_qd` only | change the example: write `joint_q`/`joint_qd` and run FK |
   | shapes overlapping at rest | add collision-filter pairs in the example |
   | non-finite contact geometry | report it with the shapes and frame from `trial.json`; check the shapes' geometry |
   | authored limits off, open loop, sinking heavy-on-light, grasp slip, warm-start bursts | apply the advice's fix with [tune-one-knob](../../steps/tune-one-knob/README.md); the guide's section 4 recipe for that case |
   | failed check, no rule fired | [tune-one-knob](../../steps/tune-one-knob/README.md): double substeps, then iterations |
   | a body vanished, nothing flagged | [render-and-grade](../../steps/render-and-grade/README.md) and [visual-review](../../steps/visual-review/README.md) before calling it lost |
   | differs from PhysX only | [check-parity](../../steps/check-parity/README.md) |

3. Re-run the whole task with the change and confirm the problem is gone and nothing else got worse (the same gates
   as before, plus the frames).
4. If the scene still fails at 32 substeps and 48 iterations, stop changing settings: it is a scene or solver
   problem. Write down the smallest reproduction (scene, settings, frame, the trial's evidence).

## Deliverables

The cause, the evidence (trial facts, frames, traces), the fix and its cost, or the reproduction for a scene or
solver problem.
