# Held-out validation of the tuning loop

**Question.** Starting from naive settings, how close does `shelf autotune` get to a hand-tuned preset,
in health and in frame time, and how many trials does it take?

## Method

- **Scenes.** 35 of these examples on skild-ai/newton `f6fc82a8b` (the examples' Newton pin), one NVIDIA RTX PRO
  6000 Blackwell per scene, each example's own world count. 29 were measured before the examples moved into this
  repository, with the advisor version noted below; the other 6 were measured in this repository with the released
  advisor.
- **Three arms per scene.**
  - *naive*: the scene's frame rate and its `model`/`collision` sections, 4 substeps, every `SolverSteel` option at
    its default (`naive/<scene>.json`);
  - *advisor*: `shelf autotune --start naive/<scene>.json --max-trials 6 --cost-trials 5`, whatever it ends with. The
    two numbers form one budget of 11 trials: at most 6 fix trials, and the cost phase gets the rest;
  - *tuned*: the example's hand-tuned `[steel]` preset.
- **Measured the same way for every arm.** Frame time: the median `example.step()` time of a full-episode trial
  (10 s, or the check's length up to 20 s) with CUDA graphs as the example captures them. Grade: an independent
  render grader that steps the episode with the GL viewer and grades non-finite state, dropped rows, speed against a
  scene bound, fall-through, bodies entering static boxes, sustained penetration, energy gain, resting jitter and
  joint-limit excess (ok / suspect / unstable; failed = did not build), plus the example's own `shelf check` at the
  same settings, plus a frozen-state check from the advisor's probe. Contact sheets of every advisor arm whose
  frame time differed from the tuned preset by 2x or more, or that was suspect, were compared with the tuned
  preset's by eye.
- **Groups.** The advisor was developed in passes, each fixing what the previous pass got wrong, so most scenes are
  not held out:
  - *rule source* (4 here): scenes whose tuning stories the rules were written from. Four more rule sources (an
    impactor, a rope winch, a pour and a projectile scene) are not in this repository and are left out;
  - *dev (run1/2/3)* (21): first seen by an earlier advisor version, whose failures there changed the rules;
  - *held-out* (4 here): added for advisor v5 (run 4). Versions v6-v9 changed the health gates and capacity routing
    (from review), the frozen-state test (from three dev scenes) and the burst tolerance (from a dev scene); these
    scenes' results were not looked at before those changes. v9 (the released rules) makes the same decision as
    v8 (run 7) on every one of the run-7 trials. Two more held-out scenes are not in this repository;
  - *held-out (final)* (6): `shapes_in_box`, `bolt_sweep`, `rods`, `bounce`, `loop_hoberman_sphere` and
    `loop_parallel_gripper`, chosen in this repository after the rules were final to replace the six scenes above
    that it does not have, and run once with the released advisor and the render audit in `examples/scripts/`.
- **Trial counts.** A trial is one full episode plus the example's check, in a fresh process. "Trials to healthy"
  counts up to the first trial the advisor itself judged healthy, build failures included; "trials" is the whole
  run, fixes plus cost candidates.

## Results

`results.md` and `results.json` hold the per-scene table (frame times in ms per frame; settings for the advisor and
tuned arms). Summary over 35 scenes:

| | naive | advisor | tuned preset |
|---|---:|---:|---:|
| graded ok | 13 | 31 | 33 |
| graded ok or suspect | 13 | 34 | 35 |
| unstable or failed to build | 22 | 1 | 0 |

- **Trials:** median 2 to the first trial the advisor judged healthy (max 5), median 7 in total with the cost
  phase (max 10). Summed trial time per scene (episode plus check): median 1.4 minutes for the whole run (max 15),
  median 0.4 minutes to that first healthy trial; 101 GPU-minutes of trials for all 35 scenes. The independent
  grading (three rendered arms and three checks per scene) came on top.
- **Frame time:** over the 34 scenes where the advisor ended ok or suspect (including the two visual suspects),
  advisor / tuned frame time is 0.56x (geomean; median 0.66x, range 0.15-1.48x). Held-out (4): 0.46x geomean;
  held-out (final) (6): 0.82x geomean, at most 4 trials to healthy.
- **Where it fell short.** `mjorg_house_of_cards` never became healthy: it stands only with Nesterov momentum plus
  warm start, a combination no rule proposes for a scene without a mass-ratio signal. `mjorg_gothic_arch` needed
  32 substeps (the preset's) and found them by blind escalation in 5 trials, 1.41x the preset's frame time because
  it stayed on the immediate response. `loop_hoberman_sphere` (8 substeps x 24 iterations
  against the preset's 4 x 64), `robot_g1` and `robot_allegro_hand` ended 1.2-1.5x slower than their presets.
- **Where it looked healthy but differed.** On two scenes the contact sheets differ from the tuned preset although
  every metric passed: `loop_fourbar` (2 x 24 instead of 4 x 32) folds flat about a second earlier, and
  `nut_bolt_sdf` (1 substep instead of 3) screws the nut down faster, the under-solved thread friction its preset's
  note describes. They are graded suspect. The advisor has no reference trajectory, so a cheaper setting that
  changes the motion without failing a metric or the check is kept.
- **Physics differences by design.** The advisor turns on authored joint limits wherever a preset leaves them off
  (Steel's default is off), so several robot arms simulate limits their tuned presets do not.

## Acceptance after this validation

The acceptance rules were tightened after this run: contacts with non-finite geometry, a trial process that failed
or timed out (whatever it wrote), an exception after the episode, and rotational grasp slip (or a health number that
stops being measured) now block a trial or a cheaper candidate; D6 joint limits are measured; the solver snapshot
records `enable_sleeping`. Replaying all 270 recorded trials through the tightened rules changes no health verdict,
fix list or cost decision. The probe and process changes act only at run time; five scenes re-run end to end with
them (`robot_ur10`, `beer_pong`, `balance_board`, `newton_cradle`, `mjorg_falling_chain`) ended on the same
settings, trial counts and frame times (within 15%) as the table.

**In this repository.** Four renamed scenes re-run with the released advisor and this repository's render audit:
`franka_plate_rack` ended on the table's settings and frame time, `anymal_d_many_worlds` on the same settings but an
articulation capacity of 768 instead of 256 (frame time within 4%);
`crane_wrecking_ball` kept 6 iterations instead of 12 (0.57x the preset's frame time against the table's 0.84x),
and its contact sheet matched the preset's; `loop_fourbar` ended on 1 substep x 24 iterations (0.21x) and again
folds flat within the first second while the preset holds its shape, the visual suspect above. The table keeps the
earlier runs for these four.

## Baseline: what the hand tuning took

The tuned presets are endpoints, not a timed baseline. The recorded search behind them, as lower bounds:

| Scene | Hand search recorded |
|---|---|
| ratio_stack (stack) | a 36-cell substeps x iterations grid, a momentum grid, then 16,384-world ladders with 3 repeats |
| overlap, torsion, linkages | found by a render audit of every scene, then 2-3 sizing runs each |
| the impactor, pour and rope scenes (not in this repository) | 20-36+ full-episode runs each: grids over substeps, iterations, baumgarte, relaxation, warm start, momentum and anchors, 64-world repeats |

That search also optimized throughput at thousands of worlds, which this loop does not; it is a documented
baseline, not an equal-budget comparison.

## Repeatability

On the 35 scenes of run 7 (29 of them in the table), two runs with rule sets that differ only in a burst-count
tolerance ended on the same settings for 19 of 35 scenes. The differences were not split between that change and run-to-run noise (a chaotic collapse, a 3% speed
bar); the tuned presets' frame times, which no rule touches, moved 0.95-1.30x between the two runs. Treat single-run frame-time ratios within about
25% as ties, and repeat a cost decision that matters.
