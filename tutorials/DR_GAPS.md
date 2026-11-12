# Domain randomization: what Steel does not cover yet

These cases have no tutorial because Steel either does not support them at runtime or the solver is
not the right place for them.

## Needs a new solver (rebuild instead of notify)

- **Turning a solver feature on or off.** Options such as `enable_joint_friction`,
  `enable_joint_velocity_limits`, `enable_joint_limits`, `friction_combine`, `restitution_combine`,
  `drive_mode` and the row capacities are fixed when `SolverSteel` is constructed. Randomize the
  values those features read, not whether the features exist.
- **Restitution with sparse mass factors.** A solver that uses sparse mass factors raises
  `ValueError` when a shape notification gives a shape positive restitution. Construct the solver
  with the restitution you will need.
- **Making a kinematic body dynamic.** A kinematic free body that was removed from the dynamics at
  construction cannot become dynamic again.

## Not shown or not yet validated

- **Collision geometry and offsets.** Shape scale, transforms, contact margin and gap are covered by
  `SHAPE_PROPERTIES`, but randomizing them changes contact generation and is not demonstrated here.
- **Swapping shapes between variants** (`Model.select_body_variants`, `SHAPE_TYPE_AND_SOURCE`) is
  experimental and is meant for resets, not every step.
- **Per-world friction on a shared ground plane.** A ground plane added to the global world is one
  shape with one coefficient. Give each world its own floor shape instead (see the terrain friction
  tutorial).
- **Contact stiffness and damping** (`shape_material_ke`, `shape_material_kd`): Steel's default
  rigid-contact rows are constraints and do not read these arrays, so randomizing them has no effect
  there. The experimental opt-in `contact_compliance` route for hydroelastic contacts has its own
  stiffness and damping inputs and is not covered here.
- **CUDA graph capture of every notification.** Joint, joint DOF, body inertial and shape
  notifications can be captured. Body (kinematic flag) notifications, sleeping, and shape
  notifications on a solver with sparse mass factors and restitution enabled read data on the host
  and must run outside capture. A captured notification also skips the eager validation of joint
  friction and rolling or torsional friction coefficients; notify eagerly to get those checks.

## Outside the solver

- **Actuator latency and delays**, **observation noise** and **action noise** belong in the
  environment or actuator model that feeds `Control`, not in the physics solver.
- **Explicit (framework-side) actuator models** keep their own gains; randomize those in the
  actuator model, because the simulator's drive gains are usually zero in that setup.
