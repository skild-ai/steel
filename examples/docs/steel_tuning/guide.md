# Steel tuning guide

How to pick settings for Newton's Steel solver (`newton.solvers.SolverSteel`) so a scene is correct first and
fast second, and how to check that it is. Every number here was measured; section 11 says on what.

The guide has two halves:

- **Sections 1-5: getting a scene right.** A fast decision procedure (symptom, likely cause, setting to try, how
  to verify), the verification loop, and recipes for the cases that break: dropped rows, start states, heavy
  bodies on light ones, chains and loops, grasps and carried contacts, warm start, torsion pads.
- **Sections 6-9: making it fast without breaking it.** What each setting costs, an option reference, pitfalls,
  and the advisor that automates the loop.

**Which Newton.** Steel is in Newton's `newton.solvers`. Two builds are in use and they spell two options
differently:

| Option | skild-ai/newton `f6fc82a8b` (the examples' pin; these numbers come from it) | upstream Newton main (2026-11) |
|---|---|---|
| sweep order | `solve_mode="serial"` (default) or `"colored"` | `solve_mode="matrix_free"` (default) |
| colored propagation | `solve_mode="colored"` + `articulated_contact_response="propagation"` | `articulated_contact_response="propagation_colored"` |
| Nesterov momentum | `nesterov_momentum=True` | not yet (open PR newton-physics/newton#5229) |
| contact adhesion | `steel:adhesion` shape attribute | not yet |

On `f6fc82a8b` the old names still work and warn (`matrix_free` maps to `serial`, `propagation_colored` to colored
propagation, `split` runs `serial`). Use the spelling your installed `SolverSteel.__init__` accepts; the
advisor detects it. Code citations below are to `newton/_src/solvers/steel/solver_steel.py` (`ss:`) at
`f6fc82a8b` unless they say otherwise.

**Hardware.** One NVIDIA RTX PRO 6000 Blackwell per measurement.

## Contents

1. [Quick start](#1-quick-start)
2. [The verification loop](#2-the-verification-loop)
3. [Symptom, cause, setting, check](#3-symptom-cause-setting-check)
4. [Recipes](#4-recipes)
5. [What not to do](#5-what-not-to-do)
6. [Cost of each setting](#6-cost-of-each-setting)
7. [Option reference](#7-option-reference)
8. [Pitfalls](#8-pitfalls)
9. [Using the advisor](#9-using-the-advisor)
10. [Validation: how much faster is it to tune with the advisor](#10-validation)
11. [Sources and provenance](#11-sources-and-provenance)

---

## 1. Quick start

Start a new scene from these, then run the loop in section 2.

| Setting | Start with | Why |
|---|---|---|
| `row_watermark` | `True` while tuning | Costs almost nothing; without it you cannot size capacities (section 4.1). |
| row capacities | measured, 1.3-2x the requested high-water mark | Defaults are 32 articulation rows, 512 free-body rows, and their sum (544) for propagation. Four of 105 audited scenes dropped rows at default capacities; three of them showed nothing on video. |
| `enable_joint_limits` | `True` if the model authors limits | Steel's default is `False`: authored limits are not simulated at all. |
| `solve_mode`, `articulated_contact_response` | serial + immediate for articulated robots and small scenes; colored + propagation for piles of free bodies | Colored propagation won on every large free-body pile timed and lost on robots and small scenes (section 6.2). Loop-closing joints need immediate. |
| substeps x iterations | Steel's 12 iterations and the substeps your scene's time step needs | Then trade: a convergence-limited scene wants the fewest substeps with more iterations; a time-step-limited one wants substeps (section 6.1). |
| `friction_anchor_beta` | `None` (anchors, 0.2) if anything is held or pinched; `0.0` (point friction) otherwise | Anchors hold carried objects that point friction plus warm start loses (section 4.6); point friction is 17-25% cheaper where nothing is held. |
| `contact_warmstart` | `False` | It needs contact matching, is rejected with sleeping, torsion and coupling, and on carried contacts it rarely matches (section 4.7). Turn it on only where you measured it helps. |
| `baumgarte` | `0.2` (default), at most `0.5` | 0.5 pulls bodies a heavy load pressed in back out; 0.8 launched bodies (section 4.4). |
| CUDA graphs | on | Without them launch overhead dominates small scenes (section 8). |

Then: run the whole episode at these settings, read the row watermarks and `check_constraint_capacity()`, run
the scene's own task check, look at it, and fix the first failing item of section 3.

---

## 2. The verification loop

Most wrong Steel settings look fine for the first second. A setting is checked only when all of these hold:

1. **The settings you ship, not cheaper test settings.** A coupled cable scene's quick check overrode the
   preset to 1 substep and 1 iteration; at those settings it went non-finite at 2.4 s, while the real preset
   stayed finite for the full 10 s. A check that changes settings tests something else.
2. **The whole task, at the load.** Instability shows when the load arrives: an impact, a grasp, a pour, a
   stack settling. A 2-frame check never reached the grasp; a 150-step probe ended before a 3 s pinch.
   Run at least the full scripted episode (the advisor runs 10 s, or the check's own length up to 20 s).
3. **No dropped rows.** Construct the solver with `row_watermark=True` and call
   `solver.check_constraint_capacity()` after the run (it raises on any loss, ss:8617). Any overflowing
   world-step is a failed run, whatever the video shows: dropped rows left a four-bar linkage's clip
   byte-identical and a pinch scene's unchanged, while the same bug made a box stack wobble.
4. **The task's own test.** Finite state is not success: most dropped start states went unnoticed because the
   scenes' checks only tested finiteness.
5. **Metrics and a look.** Non-finite state, peak and sustained penetration, peak speed against a scene bound,
   joint-limit excess, loop-closure gap, energy gain in a passive scene, resting jitter, grasp slip. Then look
   at frames across the episode (a contact sheet of 12 is enough to start).
6. **At the world count you run.** Rare failures need many worlds: a pour scene passed in 1 world and failed
   in 64; warm-start velocity bursts hit 14 of 256 worlds. A strict bar is "every world passes, in three
   repeats" (a 99.99% bar allows one failure at 16,384 worlds and none below 10,000).
7. **Motion, and settings that reached the solver.** A scene whose state stopped advancing passes every metric
   above: three scenes froze at 1 substep (an example that swaps two states inside a captured CUDA graph needs an
   even substep count) and a render grader scored all three healthy. Check that bodies move (a kinetic-energy
   series that never changes is the tell), and read back the solver's options: an example that builds its own
   solver ignores preset keys, so a "fix" can be silently absent.
8. **Compare before you blame Steel.** In one scene, a humanoid landing on an ant pushed a joint 0.35 rad past its
   limit under Steel, 0.43 rad under MuJoCo and 1.2 rad under Kamino on the same metric: a scene transient, not a solver
   fault.

**Looking at it.** Pitfalls from reviewing 100 scenes:

- A fixed camera hides things: bolts "vanished" from a tray while every one of them had landed in a bin behind a
  deck the camera could not see into. Before calling a body lost, find it in the poses, and count pieces the
  way the task's own test does (piece centres, not body origins).
- A "body inside static geometry" test on centres of mass gives false positives on visual-only boxes and
  boxes of other worlds, and misses a body that sank during the first frame unless the baseline is taken before
  the first step.
- Report instantaneous peak speeds, not speeds from frame-to-frame pose differences (29.5 m/s vs 23 m/s on
  the same run).
- Viewer-only breakage (an example reading a renamed viewer field) does not show in a headless check.

**Stop rule.** A setting is accepted when one full-episode run at the settings you ship has no dropped rows, no
non-finite state, passes the task check, shows no metric past its bound, and looks right. A cheaper setting is
accepted only if it also keeps penetration, speed, joint-limit and loop-gap numbers near those of the first
accepted setting (the advisor uses 1.5x plus a small floor) and is at least 3% faster.

---

## 3. Symptom, cause, setting, check

Work top to bottom: an earlier row makes the later rows' numbers meaningless.

| Symptom | Likely cause | Try | Verify | Cost |
|---|---|---|---|---|
| The solver will not build: `Loop-closing joints are not supported with articulated_contact_response=...` | loop joints under propagation | `articulated_contact_response="immediate"`, serial | builds | serial |
| `contact torsion ... requires contact_torsion_device=True`, or `Contact torsion graph capture requires contact_torsion_device=True` | shapes author `steel:min_torsional_patch_radius` | `contact_torsion_device=True` (and `velocity_iterations=0` with propagation) | builds; torsion rows fit (section 4.8) | small |
| `nesterov_momentum requires solve_mode='serial' ...` | momentum outside the serial immediate sweep | serial + immediate | builds | see 6.4 |
| `SolverSteel has no argument(s) [...]` | option renamed or removed in your Newton | drop it, find its replacement | builds | - |
| Kernels fail to build with very large capacities under immediate | more rows than one world's immediate kernels hold | colored propagation, size `propagation_constraint_max` | builds, no overflow | often faster on piles |
| `check_constraint_capacity()` raises; "overflow ... dropped rows" warning; `*_overflow_world_steps > 0` | row capacity too small | size from the requested mark (section 4.1) | rerun: zero overflowing world-steps | +0-22% (dropped rows now solved) |
| contact count at the contact buffer size | contact buffer full | raise the pipeline's `rigid_contact_max` | high-water below capacity | small |
| A start pose or velocity is lost on the first step | written to `body_q`/`body_qd` only; Steel integrates `joint_q`/`joint_qd` | write the start into the state's joint coordinates (section 4.2) | joint coordinates near the requested start one step later | none |
| Mechanism fights its drives from frame 1, energy where none should be | shapes of one mechanism overlap at rest | collision-filter pairs (section 4.3) | energy and peak speed drop | none |
| Joints far past their limits | `enable_joint_limits=False` (Steel's default) | `enable_joint_limits=True` | limit excess small | a few rows per limited DOF near a limit |
| Light bodies pressed into the ground or another body by a heavy one | Gauss-Seidel convergence at a large mass ratio (above ~1000:1, a heavy impactor) | 2x substeps, 2x iterations, `baumgarte=0.5` (section 4.4) | no light body below rest by more than half its size | 2.6-2.8x frame time |
| A resting stack of increasing masses slumps or topples (~100-1000:1) | too few sweeps per frame for the load to reach the bottom | Nesterov momentum; then a sweep budget, spent as iterations in the fewest substeps (section 4.5) | the stack stands the whole check | small to 1.5x |
| A rope, chain or driven mechanism goes non-finite or fails its check at low substeps; more iterations do not help | time-step limited dynamics | more substeps, few iterations (section 4.5) | the check passes at the world count you run | ~linear in substeps |
| A loop-closing joint opens (anchor gap > ~1 mm), a four-bar tilts out of plane | loop rows not converged | 2x iterations (four-bar needed 32), then substeps | loop gap below 1 mm for the episode | ~linear |
| A held or carried object slips or twists in the grasp | point friction plus warm start on moving contacts (section 4.6) | `friction_anchor_beta=0.2` (or `None`), `contact_warmstart=False` | slip < 10 mm / 10 deg over the hold; task check at >= 64 worlds | none measured (faster on the pour scene) |
| Grasp slips with anchors on | friction rows not converged | 2x iterations | as above | ~linear |
| Rare launches / bursts (a body gains > 2 m/s in a frame) with warm start on | warm-started impulses on deeply wedged contacts | `contact_warmstart=False`, or keep it with `contact_regularization=0.01` | zero bursts over >= 256 worlds | none |
| Bodies launched after a hard contact, with `baumgarte` raised | position correction too strong | `baumgarte <= 0.5` | peak speeds against a reference solver | none |
| Deep resting overlap of fast small objects against thin supports | travel per substep far beyond the detection margin | more substeps, or a larger margin on the thin support | penetration below half the thinner shape | ~linear |
| The task check fails and none of the above fired | unknown | double substeps, then iterations, one at a time | the check passes | ~2x per step |

The advisor (section 9) detects every row of this table it can from one run: the construction errors, the
capacity overflow, the start state, the authored-but-off limits, the contact-island mass ratio, the loop gap,
grasp slip, warm-start bursts and match rates, and the failed check.

---

## 4. Recipes

### 4.1 Row capacities

**The families.** Steel gives each world fixed row capacities (ss:2008-2071, ss:2186-2203):

| Family | Option (default) | Holds |
|---|---|---|
| articulation (dense) | `articulation_constraint_max` (32) | joint limits, mimic and loop rows, joint friction, drive rows with `drive_mode="rows"`, dense torsion rows; under the immediate response also every contact of an articulated body; under propagation, contacts within one articulation |
| free body | `free_body_constraint_max` (512) | free-body contacts under the immediate response; under propagation only free-body velocity-limit rows |
| propagation | `propagation_constraint_max` (`None`: the other two summed) | every contact under propagation except same-articulation ones, free bodies included, plus one spare row per positive-torsion-radius contact |

`propagation_constraint_max` may only be set with the propagation response (`ValueError` otherwise, ss:2901). A
frictional contact takes 3 rows (normal plus two tangents).

**Measure.** Construct with `row_watermark=True`, run the full episode at the settings you ship, then read
`solver.constraint_row_watermarks()` (ss:8539). Per family it returns `*_high_water` (retained rows, capped at
the capacity), `*_raw_high_water` (requested rows), `*_dropped_contact_rows_high_water`,
`*_overflow_excess_high_water` and `*_overflow_world_steps`, plus `contact_high_water`. Gate with
`solver.check_constraint_capacity()`, which raises if any world lost rows or contacts since the last reset.

**Size.**
- Never size from `*_high_water`: it stops at the capacity (an overflowing box stack read 544).
- While a family overflows, `*_raw_high_water` undercounts: the dropped contacts change the dynamics. The same
  20-box stack requested 1,009-1,020 rows at the default 544 and 1,038 with room; a 20-substep projectile scene
  705 against 713. Raise an overflowing family to about 2x its requested mark, rerun, then keep 1.3x headroom
  over the mark measured without overflow.
- Measured fixes: the box stack 1,038 -> 1,344 propagation rows; the projectile scene 713 -> 896; a pinch grip
  42 -> 64 articulation rows; a four-bar linkage 40 -> 64 (overflowing from t = 0.68 s); a card house 658 ->
  1,024 (it collapsed at 5.5 s); a trebuchet 1,771 -> 3,072 at the wall impact; a ragdoll 56 -> 128; a
  480-domino spiral 7,179 -> 12,288 (it exploded at 80 m/s at the default).
- Robots need more than 32 articulation rows whenever limits or self-contacts are on: four feet overflowed 32;
  measured needs ranged from 34 to 244.
- Very large immediate-response capacities stop the kernels from building (a 168-piece pour at 41,760 free-body rows);
  put such scenes on colored propagation, which ran the same pour at 16,384 propagation rows.

**Cost.** Raising an overflowing capacity costs time because rows that were skipped are now solved: the box
stack went from 4.84 to 5.89 ms per frame (+22%); the linkage +0%; the projectile scene +6%.

### 4.2 Start states

Steel's state is `State.joint_q` and `State.joint_qd` (ss:1468-1475). Free bodies have a free joint (`add_body`
adds one) with `joint_qd = (v_com, omega)` in the parent-anchor frame. Bodies on no joint at all pass
`body_q`/`body_qd` through and enter contacts at zero velocity (ss:1544-1547).

So a start written only to `body_q` or `body_qd`, or written to `model.joint_q` after `state_0` was created,
is lost on the first step. Seen on: a dumbbell that should spin at 20 rad/s (joint_qd was 0, so it never spun),
a 100 m/s projectile that never moved, a linkage whose requested start angles became 0. Finiteness checks pass
on all of them.

Fix: put start coordinates on the builder (`joint_q`, `joint_qd`) before `finalize()`, or compute them with
`newton.eval_ik(model, state, state.joint_q, state.joint_qd)` and then `newton.eval_fk(...)` into the same
state. Check: compare `state.body_q`/`body_qd` with `eval_fk` of the state's joint coordinates before step 0
(the advisor does; float32 FK of an exported pose differs by about 2 mrad, real misses were 0.8 rad and
20 rad/s), and one step later the joint coordinates should be near the requested start.

### 4.3 Shapes that overlap at rest

Shapes of one mechanism that overlap at rest and are not filtered fight the drives from the first frame. A
wrist capsule 21 mm inside its own hand put 65 J and 7.8 m/s into a 0.43 kg box; with four filter pairs, 1.3 J
and 0.41 m/s. A 1 mm overlap with nearly tangent normals held a toaster carriage 38 mm above its limit, so it
never latched. No solver setting fixes this: add collision-filter pairs for shapes that should never touch.
Detect it by colliding once at t = 0 and flagging overlaps above ~1 mm between shapes of one articulation that
are not parent and child.

### 4.4 Heavy bodies on light ones: an impactor

A 2 m wrecking ball of 100x density rolling over a pyramid of 0.8 m cubes (about 6,500:1) presses the cubes it
rolls over into the ground: Gauss-Seidel passes the ball's load through each cube's rows in turn, and at a few
iterations the cube-ground rows never see it. Three pyramids at 100 fps, colored propagation, lowest cube centre
(rest height 0.4 m) and p95 penetration over 10 s; frame times on one NVIDIA RTX PRO 6000 Blackwell, 1 world,
with XPBD on the same scene and GPU as the reference:

| Settings | Frame ms | Lowest cube z [m] | p95 penetration |
|---|---:|---:|---:|
| 2 substeps x 12 iterations | 1.50-1.55 | -0.31 / -0.23 | 45-48 cm |
| 2 x 24 / 2 x 48 | 2.08 / 3.04 | -0.11 / 0.15 | 41 / 22 cm |
| 4 x 12 / 8 x 12 / 16 x 12 | 3.01 / 5.81 / 13.25 | 0.09 / 0.27 / 0.385 | 30 / 8 / 2.5 cm |
| warm start + matching (2 x 12) | 1.66 | -0.22 | 61 cm |
| relaxation 1.3 | 1.59 | -0.365 (51 m/s ejections) | 46 cm |
| velocity_iterations 2 | 1.67 | -0.33 | 43 cm |
| Nesterov (serial immediate) 12 / 24 iterations | 8.23 / 16.97 | 0.12 / 0.29 | 26 / 16 cm |
| baumgarte 0.5 / 0.8 (2 x 12) | 1.53 / 1.45 | -0.02 / 0.15 (7-8 bodies left the scene) | 15 / 14 cm |
| **4 x 24, baumgarte 0.5** | 3.90-4.65 | 0.20-0.37 | 3.5-10.6 cm |
| 4 x 16, baumgarte 0.5 | 3.26-3.40 | 0.18-0.21 (12-13 left the scene) | 5.5-11 cm |
| 6 x 16, baumgarte 0.5 | 5.08-5.13 | 0.30-0.34 | 3-8.5 cm |
| XPBD reference (10 substeps) | 1.08 | 0.393 | 1.2 cm |

Recipe: double substeps and iterations together and raise `baumgarte` to 0.5; repeat once if light bodies still
sink by more than half their size. On the one-pyramid scene that was 2.7-2.8x the frame time. Expect residual
penetration well above XPBD's and faster ejections (on this scene, fastest body 26-41 m/s against XPBD's 21-24): it is a
partial fix. Do not use relaxation above 1 (it made sinking worse), `baumgarte` above 0.5, or warm start
(needs contact matching, no gain); Nesterov helped but forces the serial immediate sweep, 5-29x the cost here.

### 4.5 Stacks, chains and ropes: sweeps or time step

Two different limits look alike ("more substeps fixed it"), and they want opposite settings.

**Convergence-limited: a resting stack.** Six cubes, each 3x the one below (243:1 top to bottom); the check
fails if a cube drops 5 cm. On a 6x6 grid of substeps {1,2,3,4,6,8} x iterations {4..24} at 256 worlds:

- without momentum only 8 x 24 passed (192 sweeps per frame); every cell under 192 sweeps dropped the stack;
- with `nesterov_momentum=True` (serial immediate) the pass boundary was about **96 sweeps per frame,
  whatever the split**: 4 x 24, 6 x 16 and 8 x 12 passed, 8 x 8 and 6 x 12 failed;
- at 16,384 worlds 1 x 88 ran 616k world-frames/s, 4 x 24 449k, the original 8 x 16 307k: once the requirement
  is a number of sweeps, spend it as iterations in the fewest substeps. With momentum off, the top cube ended
  at 0.355 m instead of 0.845 m.

**Time-step-limited: a rope on a winch.** A robot arm turns a crank that winds an articulated rope and lifts a
bucket. At 24 sweeps per frame, 4 x 6 passed every world, 3 x 8 99.6%, 2 x 12 96%; at 1 substep it went
non-finite at every iteration count. Here substeps buy what iterations cannot. With momentum, under-relaxation
(`relaxation=0.6`) and warm start, 2 x 8 passed every world at 8,192 worlds and was the fastest cell.

**Why fewest substeps usually wins.** Fitting `frame_ms = s (a + b i)` on 24 contact scenes, one substep's
fixed cost `a` (collision-independent solver setup, kinematics) equals 7-60 iteration sweeps `b`, median 18
(stack 24, rope 17). Winning cells used 1-2 substeps in 13 of 18 scenes.

**How to tell which limit you hit.** Halve substeps at the same iterations, and halve substeps at twice the
iterations. If the second passes, the scene is convergence-limited: keep going toward fewer substeps. If neither
passes, it is time-step-limited: keep the substeps and try fewer iterations instead.

**Loops.** Loop-closing joints are solved only with the immediate response (propagation raises, ss:3029). Their
rows converge like any others: a four-bar at 12-16 iterations let out-of-plane tilt grow until it fell (32
held); a closed leg linkage held a 0.10 mm loop gap at 24 iterations and 0.052 mm at 50; a test mechanism opened
0.6-1.7 mm at 10-15 substeps. Watch the loop-anchor gap over the episode.

### 4.6 Grasps and carried contacts

A two-finger pinch lifts a carton, pours 168 letters into a bowl and swings back. With point friction
(`friction_anchor_beta=0`) and warm start on, **no world** of 64 held the carton through the return swing. What
fixed it, at 3 substeps x 8 iterations, colored propagation:

| Change | Worlds holding the carton |
|---|---:|
| baseline (point friction, warm start on) | 0 / 64 |
| warm start off alone; warm-start decay 0.5; match threshold 2, 5 or 10 mm; 6 substeps | 0 / 64 |
| 16 / 24 / 32 iterations | 0 / 24 / 53 of 64 (32 is ~2x the cost) |
| 8 substeps x 16 iterations, warm start off | 63 / 64 (~3x the cost) |
| anchors 0.1 | 34 / 64 |
| **anchors 0.2, warm start off** | 64 / 64 four times; 256 / 256 |
| anchors 0.2, warm start on, `contact_regularization=0.01` | 256 / 256 |

The fix was 1.10x *faster* than the broken setting (26.3 against 28.9 ms per frame at 64 worlds). Anchors 0.2
is Steel's default (`friction_anchor_beta=None` resolves to 0.2 unless `contact_compliance` or a
non-`fixed_normal` friction mode selects point friction, ss:2667-2688).

**Why.** Anchors carry each contact region's tangential displacement across steps and add
`beta * displacement / dt` to the tangent rows; their history is matched by body pair and geometry,
independently of collision contact matching (ss:2284-2297). Warm start instead depends on
`CollisionPipeline` contact matching, which pairs a new contact with a previous contact of the same shape pair
whose world-space midpoint lies within `contact_matching_pos_threshold` (default 0.5 mm) and whose normal
agrees (collide.py:1826; contact_match.py). A contact carried at speed v moves v x dt per substep; once that
passes the threshold, matches fail often. In a pinch repro, lifting at 0.05 m/s kept 100% of contacts matched
and 0.6 deg of twist; at 0.1 m/s 28% matched and the object twisted 24.5 deg; at 0.2-0.5 m/s 0.5-0.6% matched.
On the pour scene the carton's contacts matched 4 of 4 while still and 0 of 4 from the lift. (The matcher
searches all previous contacts of the pair, so treat threshold / dt as a heuristic speed, about 0.09 m/s at
3 substeps of a 60 fps frame, not a cutoff.)

**Matching also orders the contacts.** Any matching mode sorts contacts deterministically (collide.py:2064).
A sorted stream gives the same finger priority every sweep: a static hold twisted 4.5 deg unsorted and 27.8
deg sorted. Turning warm start on for a grasp brings the sort even where matches fail.

**Matching in the body frame** (matching contact midpoints in the bodies' frames) fixed the pinch twist in a
prototype (25.1 -> 0.68 deg, +5% collision time, under 1% per step) but is not in the Newton build these examples
pin (`f6fc82a8b`); with anchors on it added nothing on the pour scene.

**Recipe.** Anything held or carried: anchors (`None` or 0.2), warm start off. If it still slips, double the
iterations. Check slip relative to the holding link over the whole hold, and run the task check at 64 or more
worlds: one world passed where 64 failed.

### 4.7 Warm start

`contact_warmstart=True` starts each matched contact from last step's impulse.

- **It needs contact matching** on the contacts buffer, or the step raises (ss:7370).
- **It is rejected with** sleeping (sleeping.py:25), contact torsion (contact_torsion.py:91), friction
  position iterations (ss:2802) and coupled solves (ss:4133-4148).
- **On carried contacts it rarely matches** (section 4.6).
- **On deeply wedged resting contacts it can launch bodies.** 3.4 mm letters landing in a bowl end 7-11 mm deep
  in its 12 mm bottom; with warm start, 14 of 256 worlds produced fast letters (peak 70 m/s) and 3 fell through
  the table. Warm start off, or `contact_regularization=0.01` with warm start kept, removed every burst. The
  regularization's resting cost is about `g a dt^2 / baumgarte`, here 0.015 mm (ss:2298-2313).
- **Where it helped:** a house-of-cards scene stands only with momentum plus warm start (every
  colored setting, 4-32 substeps and 16-64 iterations, lost the apex within 2 s); the rope winch's fastest
  passing cell used it; a toy piano used it.
- **Matching sort keys** alias above 2^20 shapes on builds without newton-physics/newton#5233.

Rule: off by default; on only for a scene where you measured that it holds something the cold start does not,
and never for a grasp without anchors.

### 4.8 Torsion pads

A positive `steel:min_torsional_patch_radius` on a shape turns on one angular row per contact group of an
articulated body, bounded by `r (mu N - sliding)` (ss:2418-2470). The radius is an effective footprint (`2R/3`
for a uniform disk of radius `R`).

- With the propagation response it needs `contact_torsion_device=True` and `velocity_iterations=0` (ss:2935-2945);
  CUDA graph capture needs `contact_torsion_device=True` with either response. It requires
  `contact_warmstart=False`.
- Dense torsion rows count against `articulation_constraint_max` after the world's other dense rows; a
  propagation-routed contact reserves its torsion row in the propagation family. A world that dropped dense rows
  gets no torsion rows at all, so size capacities first.
- Turning torsion off is a physics change, not a free speedup; MuJoCo's default condim 3 has no torsional friction.

### 4.9 Joint limits, damping, adhesion and mesh contacts

- **Limits.** `enable_joint_limits` defaults to `False`.
  A fixed `joint_limit_activation_gap=0.1` builds limit rows only near a limit and was 8-40% faster than the
  speed-aware default on 8 scenes. A joint that can cross the gap in one step misses its row: keep the gap above
  2 x max |qd| x dt.
- **Joint damping** is explicit on the build measured: it is unstable when `dt * damping / m_eff > 2` (light
  links next to strong damping). Keep it below that per DOF (raise armature or substeps). An implicit damping
  integration is proposed upstream (newton-physics/newton#5010).
- **Adhesion** (`steel:adhesion`, `f6fc82a8b` only): pull per shape pair within the pair's gap, friction bounded by
  `mu (N + A)`. Any adhesive shape forces the generic dense path (+14% on a legged scene); it is rejected with
  sleeping, compliance, regularization and contact torsion.
- **Box against a non-convex mesh.** On the build measured, a box against one-sided triangles of a non-convex
  `GeoType.MESH` can report contacts centimetres deep and sideways (65 mm on CPU, 99.9 mm on CUDA, against
  0.5 mm before the change that introduced it; newton-physics/newton#5240). Convex hulls, SDF meshes,
  heightfields and box-box are unaffected. Use convex hulls or check penetration on such pairs. The fix
  (newton-physics/newton#5241) leaves one case open: a thin box nearly parallel to a one-sided triangle can get a
  contact about a metre deep ([#5256](https://github.com/newton-physics/newton/issues/5256); a guard is proposed in
  [#5258](https://github.com/newton-physics/newton/pull/5258), under review). On a build with #5241 but not #5258,
  check thin boxes against mesh triangles.

---

## 5. What not to do

Fairness when comparing solvers, and safety when tuning: a setting may change how Steel solves the physics,
never which physics it solves.

- Do not turn off joint limits, joint velocity limits, joint friction, springs, restitution, torsion, friction
  anchors that the scene relies on, collision or self-collision to win time. Report such runs as diagnostics.
- Allowed: substeps, iterations, solve mode and response, row and contact capacities, broad phase, warm start,
  regularization, sleeping thresholds, skipping contacts between two static or world-welded bodies (MuJoCo skips
  them too), fewer speculative contacts within the travel bound.
- Compare backends at matched time steps, on the same GPU and the same stack, against a strict bar (every world
  of three repeats).
- Do not size capacities from a run that overflowed, and do not shrink them below 1.3x a non-overflowing mark.

---

## 6. Cost of each setting

### 6.1 Substeps and iterations

- One substep's fixed cost equals 7-60 iteration sweeps (median 18) across 24 contact scenes (section 4.5).
- Winning cells used 1 substep in 5 scenes, 2 in 8, 3 in 3, 4 in 1 and 6 in 1. Scenes' original settings were
  mostly 4 or 8 substeps, so the usual throughput lever is fewer substeps, with more iterations where the scene
  is convergence-limited.

### 6.2 Solve mode and response

- **Colored propagation** won on piles of free bodies: card house, jenga, dominoes, a bolt sweep, the pour,
  a drawer with loose objects, a toy train.
- **Serial immediate** won on articulated and small scenes: air hockey, a balance board, a bicycle linkage, a
  plate rack, a ragdoll, the mass-ratio stack, a toaster, a toy piano, the rope winch, a trebuchet. Measured
  losses for colored: 3-7x slower on scenes of humanoids and ants (their multi-DOF joints take generic tree
  kernels), 3.6x on a quadruped; a
  humanoid fell under colored at 1 x 4 and 2 x 2 and needed 1 x 8.
- Colored is bound by the busiest world and the body with the most contacts. That body's contact degree is a lower
  bound on the colour count; the greedy first-fit colouring can use more (up to `2 * degree - 1`), depending on the
  order it meets the contacts. Colours past 512 spill into a serial tail.
- On the serial sweep each world is one block, so step time follows the busiest world's rows, not the mean.

### 6.3 Friction and contacts

| Change | Measured |
|---|---|
| point friction (`friction_anchor_beta=0`) where nothing is held | 17-25% faster on legged, sensor and policy scenes |
| anchors instead of point friction plus warm start on a carried grasp | 1.10x faster and correct (section 4.6) |
| `contact_gap_gate=0` where speculative rows never engage (omniwheel rollers) | halved the frame time |
| a fixed limit activation gap of 0.1 | 8-40% on 8 scenes |

### 6.4 Accuracy fixes

| Fix | Cost |
|---|---|
| sizing capacities that overflowed | +0-22% (the rows are now solved) |
| heavy impactor: 2x substeps x 2x iterations + baumgarte 0.5 | 2.6-2.8x |
| `nesterov_momentum` | +7-16% per sweep (legged +15%, contact D6 +7.1%), and it bypasses the sparse mass, sparse-diagonal, local and paired-factor fast paths; off is bit-identical |
| `contact_regularization=0.01` | ~0.015 mm extra resting penetration at 60 Hz x 3 |

---

## 7. Option reference

Defaults are `SolverSteel.__init__` at `f6fc82a8b` (ss:2008-2071). No `**kwargs`: an unknown name raises.
`baumgarte`, `relaxation`, `warn_constraint_overflow`, `velocity_drive_mode`, `contact_warmstart_decay`,
`restitution_velocity_threshold`, `contact_speculative_scale`, `contact_gap_gate` and `contact_friction_gap_max`
can change between steps; everything else is fixed at construction. Two limits apply:

- `contact_friction_gap_max` keeps the category it was constructed with. `None` and `+inf` (friction rows on every
  contact) and a finite value or `-inf` (a gap test) are separate categories; assigning across them raises
  `ValueError` (ss:1422). A default solver cannot become frictionless by assignment: construct a new solver with
  `contact_friction_gap_max=-inf`.
- A captured CUDA graph keeps the values it was captured with. Capture the step again after changing any of these.

| Option | Default | What it does | Tune when |
|---|---|---|---|
| `solve_mode` | `"serial"` | Gauss-Seidel order: one fused per-world kernel, or per-step colouring of propagation rows | colored for piles (requires propagation) |
| `articulated_contact_response` | `"immediate"` | dense `H^-1 J^T` rows, or 6-wide body rows through the articulated-body factorization | propagation for colored; immediate for loops, momentum |
| `iterations` | 12 | sweeps per substep | convergence-limited scenes, loops, grasps |
| `baumgarte` | 0.2 | position correction of contact and limit rows | up to 0.5 for sinking under heavy loads |
| `relaxation` | 1.0 | SOR factor | 0.6 helped a momentum rope; above 1 hurt a heavy impactor |
| `articulation_constraint_max` | 32 | dense rows per world | from watermarks |
| `free_body_constraint_max` | 512 | free-body rows per world (immediate) | from watermarks |
| `propagation_constraint_max` | `None` (sum) | propagation rows per world | from watermarks; propagation only |
| `row_watermark` | `False` | high-water counters for `constraint_row_watermarks()` | on while tuning |
| `enable_joint_limits` | `False` | limit rows | on when limits are authored |
| `joint_limit_activation_gap` | `None` (speed-aware) | distance from a limit at which its row exists | 0.1, above 2 max\|qd\| dt |
| `enable_joint_velocity_limits` | `False` | velocity-limit rows | on when the task needs them |
| `friction_anchor_beta` | `None` (0.2; 0 with `contact_compliance` or a non-`fixed_normal` friction mode) | friction anchors; 0 is point friction | anchors for grasps; 0 where nothing is held |
| `contact_warmstart` | `False` | start matched contacts from last impulse | see 4.7 |
| `contact_regularization` | 0.0 | implicit contact spring weight | 0.01 to keep warm start without bursts |
| `velocity_iterations` | 0 | velocity-only sweeps after the position solve | a stacking scene used 4 to settle; 0 with torsion under propagation |
| `enable_restitution` | `True` | restitution targets above `restitution_velocity_threshold` (0.5 m/s) | |
| `contact_torsion_device` | `False` | device torsion rows (graph capture) | on whenever torsion radii are authored |
| `propagation_color_order` | `"contact"` | colour order | leave it; time another order on the scene |
| `nesterov_momentum` | `False` | Nesterov extrapolation of impulses between sweeps, with gradient restart | resting stacks, large mass ratios, long chains (serial immediate, CUDA) |
| `contact_gap_gate` | `None` | drops rows of contacts beyond this gap | 0 for speculative rows that never engage |
| `contact_friction_gap_max` | `None` | friction rows only within this gap | `-inf` for a frictionless scene (at construction; see above) |
| `enable_sleeping` | `False` | island sleeping | settling scenes (excludes warm start) |
| `parallel_tree` | `False` | tree-parallel FK and inverse dynamics | articulations of 64+ DOFs; time it |
| `use_parallel_streams` | `False` | per-size streams, fused CRBA, the sparse-diagonal contact solve | many independent 1-DOF joints on one base |
| `reuse_kinematics` | `False` | carry kinematics between steps | time it |

**Nesterov momentum.** Each sweep after the first extrapolates impulses by `(k-1)/(k+2)` along the previous
sweep's change; it restarts when the last correction opposes the extrapolation and when friction rows switch on
(ss:16699). Velocity-limit rows are not extrapolated; contact and position-limit rows are. It requires
`solve_mode="serial"`, `articulated_contact_response="immediate"`, `row_schedule="interleaved"` and CUDA
(ss:2843-2850). Measure convergence on the scene before lowering iterations because of it: a better-converged
solve also changes contact forces, so recheck force-sensitive tasks.

---

## 8. Pitfalls

1. **Deprecated and removed names.** Shelf-style presets written for one build fail on another: an option
   disappeared between two pins without a shim (`enable_torsional_rolling_friction`). Validate settings against
   the installed `SolverSteel.__init__` signature.
2. **Cheap test settings.** See section 2, item 1.
3. **Dropped rows that look fine.** See section 4.1; a passing finiteness check said nothing.
4. **Overflow under one setting, fine under another.** Point friction takes two friction rows per contact where
   anchors share them per region; a budget that fit with anchors overflowed with point friction. Re-measure
   after any friction or response change.
5. **Odd substep counts with a captured state swap.** Examples that swap two states inside a captured graph
   freeze with an odd number of substeps per frame; keep it even there. The frozen run looked perfect: no
   penetration, no speed, no loop gap, a 6x shorter frame.
6. **One world is not a test.** See section 2, item 6.
7. **Graph-off profiles are not throughput.** Launch overhead dominates small scenes with graphs off; time
   decisions with graphs on and a warm kernel cache.
8. **Different GPUs.** Ratios move between GPUs; compare arms on one GPU in one job.
9. **Particles.** `SolverSteel` refuses models with particles; coupled examples build it on the rigid part only.
10. **Settings the example overrides.** An example that constructs `SolverSteel` itself may hard-code options
    (point friction, capacities) whatever its preset says. Compare the options you asked for with the
    constructed solver's before you conclude a setting did nothing.
11. **Thin pieces in hull supports.** Thin capsules (3.4 mm) falling at about 2 m/s onto a convex-hull bowl rest
    past half their thickness in its bottom, also with the shipped settings; a 15 mm static gap and speculative
    contacts did not reduce the depth. The task still passes. A depth metric normalized by thickness flags them; read it with the task check.

---

## 9. Using the advisor

The advisor (`common/advisor/`, behind `shelf trial`, `shelf autotune` and `shelf advise`) automates sections
2-4. The rules are plain Python on recorded facts; the probe needs Newton and Warp.

**On a shelf example** (`uv run python -m shelf ...`, or `.venv/bin/python -m shelf ...`):

```bash
# One trial: the whole episode at the given settings, plus the example's own check
python -m shelf trial crane_wrecking_ball --settings '{"sim": {"substeps": 2}, "solver": {}}' --out outputs/crane_t0

# The loop: fix until nothing is left, then try cheaper settings that stay healthy
python -m shelf autotune crane_wrecking_ball --start defaults --out outputs/crane_tune
python -m shelf autotune crane_wrecking_ball --start my_settings.json --max-trials 6 --cost-trials 5 --out outputs/crane_tune
```

`autotune` runs each trial in a fresh process. Phase 1 applies the first-priority fixes of section 3 until
none is left; when every fix of a round names a metric and none of those metrics drops by 25% (and the trial is
not healthy), the round is reverted and those fixes are not tried again. A metric missing before or after a fix
counts as no evidence of improvement. Phase 2, with whatever is left of the shared budget (`--max-trials` +
`--cost-trials` trials in all), tries, in order, colored propagation (20+ free bodies per world),
half the substeps, half the substeps at twice the iterations, a 0.1 limit gap, point friction (only when nothing
is held) and half the iterations; a candidate is kept only if it is healthy, at least 3% faster and its health
numbers stay within the bounds of section 2's stop rule. Healthy always requires a finite, complete episode with
a passing capacity check and the example's check.
`tune.json` records every trial with its settings, summary, actions and flags; `trialN/trial.json` holds the
full facts.

**In your own Newton script**, use the probe directly:

```python
from common.advisor.episode import EpisodeProbe, model_facts, start_state_facts, time_step
from common.advisor.stability import recommend_stability, summarize

solver = newton.solvers.SolverSteel(model, row_watermark=True, **my_kwargs)
start = start_state_facts(model, state_0)          # before the first step
probe = EpisodeProbe(model, solver, substep_dt, collision_dt, match_threshold=None)
probe.observe_initial(state_0)                     # before the first step: sinking and frozen-state baselines
for frame in range(num_frames):                    # the whole task
    ms = time_step(step_one_frame)                 # your frame: collide + substeps
    probe.update(state_0, contacts, ms)
facts = {"settings": {"sim": {"substeps": n}, "solver": my_kwargs}, "solver_kwargs": my_kwargs,
         "start_state": start, "model": model_facts(model, solver), "episode": probe.finish(),
         "frames_planned": num_frames, "frames_done": num_frames, "frame_dt": frame_dt, "substeps": n,
         "check": {"ok": my_task_check_passed}}
advice = recommend_stability(facts)
print(summarize(facts)); print(advice["actions"]); print(advice["flags"])
```

**What it detects.** Construction errors with a known fix (including settings an example reads from its
preset); row overflow per family (sized from 2x the requested
mark, or moved to colored propagation past 8,192 rows per world under immediate); a full contact buffer; start
states written to body arrays; authored limits left off; open loops; mass-ratio islands with sinking (impactor
recipe above 1,000:1, momentum and sweeps between 100 and 1,000:1); grasp slip (anchors, then iterations);
warm-start bursts and unmatched carried contacts; a failed check with no other cause (substeps, then
iterations); centres of mass entering static boxes; a state that stops advancing; settings the example
overrides; motion of jointed objects (handles, knobs) that a naive grasp check would call slip; deprecated
spellings; short or truncated episodes. Thresholds and costs are in
`common/advisor/stability.py`, `STABILITY_CALIBRATION`.

**What it cannot see.** Visual problems no metric catches (look at the frames), scene bugs other than start
states and first-frame overlaps, task success beyond the example's own check, and fidelity to a reference
solver.

---

## 10. Validation

How close does the loop get to a hand-tuned preset from naive settings, and how fast? Full method, per-scene table
and settings: [`advisor_validation/`](advisor_validation/) (README, `results.md`, `results.json`, `naive/`).

**Setup.** 35 of the examples, one NVIDIA RTX PRO 6000 Blackwell each. Three arms: *naive* (the scene's frame rate and model settings,
4 substeps, every `SolverSteel` option at its default), *advisor* (`autotune` from naive with a shared budget
of 11 trials: at most 6 fix trials, the rest cost trials) and *tuned* (the scene's
hand-tuned preset). Every arm is timed the same way (median frame of a
full-episode trial) and graded by an independent render grader plus the scene's own check and a frozen-state
check; contact sheets of every advisor arm far from the preset in frame time, or suspect, were compared by eye.
The rules were written from 4 of the scenes (and four others not in this repository) and corrected on 21 more in
earlier passes. 4 scenes were added for the fifth advisor version; the three later versions changed only the
frozen-state test, the burst tolerance and health gates, each prompted by other scenes or by review, and those
scenes' results were not looked at before the changes. The last 6 scenes were chosen after the rules were final
and run once with the released advisor.

| | naive | advisor | tuned preset |
|---|---:|---:|---:|
| graded ok | 13 | 31 | 33 |
| graded ok or suspect | 13 | 34 | 35 |
| unstable or failed to build | 22 | 1 | 0 |

- **Trials:** median 2 to the first trial the advisor judged healthy (max 5); median 7 in all (max 10). Summed
  trial time per scene (episode plus check, in fresh processes): median 0.4 minutes to that trial, 1.4 minutes
  for the whole run (max 15); the independent grading came on top.
- **Frame time:** advisor / tuned 0.56x geomean (median 0.66x, range 0.15-1.48x) over the 34 scenes it ended
  ok or suspect on; 0.82x on the last 6 scenes, each judged healthy by its fourth trial.
- **Misses:** a house of cards never became healthy (it needs momentum plus warm start); a gothic arch needed 32
  substeps, found by blind escalation in 5 trials. Three more ended 1.2-1.5x slower than their presets.
- **Passing but different:** two scenes passed every metric with visibly different motion (a four-bar that folds a
  second early, a nut that screws down faster at 1 substep). The loop has no reference trajectory: look at the
  result next to a trusted setting before shipping a much cheaper one.
- **For comparison,** the hand tuning behind the rule-source presets took 20-50+ recorded runs each (grids, sweeps,
  64-256-world repeats), and the dropped-row presets were found only by a render audit of every scene.
- **Repeatability:** two runs whose rules differed only in the burst tolerance ended on the same settings for 19
  of the 35 scenes of an earlier run, and the tuned presets' frame times moved 0.95-1.30x between them. The differences were not split
  between that change and run-to-run noise; treat single-run frame-time ratios within about 25% as ties.

The advisor makes frame time cheaper mostly by finding the fewest substeps a scene tolerates (section 6.1); it
also turns on authored joint limits wherever a preset leaves them off.

---

## 11. Sources and provenance

- **Newton.** skild-ai/newton `f6fc82a8b` (the revision `examples/pyproject.toml` pins), files under
  `newton/_src/solvers/steel/` and `newton/_src/sim/collide.py`, `newton/_src/geometry/contact_match.py`; upstream
  Newton main `0679304` (2026-11-04) for the spelling differences. Upstream issues and PRs cited by number are on
  newton-physics/newton.
- **Scenes.** The examples as they were in the Steel Shelf project before they moved into this repository (169
  examples then; this repository keeps the 81 whose Steel presets pass every check), one `presets.toml` per
  example with a `[steel]` table whose comments record why each setting is there. Scene descriptions above are
  generic. Those still in `examples/shelf/` are ratio_stack (stack), overlap (box stack), torsion (pinch grip),
  linkages (four-bar), card_house, trebuchet, ragdoll, domino_spiral, box_turn and toaster (overlapping shapes),
  mjorg_house_of_cards, loop_fourbar, loop_dr_legs and loop_dr_testmech. The impactor (a ball rolling over
  cube pyramids), rope winch, pour and projectile scenes are not in this repository.
- **Measurements.** Shelf scenes: full-episode runs with `row_watermark=True`, one GPU, CUDA graphs as the
  example captures them, 1 world unless a world count is given. Throughput grids: 256-world 6x6 substeps x
  iterations grids per scene, then world-count ladders up to 16,384 worlds with three repeats under the bar of
  section 2.
