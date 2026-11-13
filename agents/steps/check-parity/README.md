# Check parity

> **Does:** compares two traces of the same task step by step by name, and audits each difference down to a cause.
> **Use in:** port-from-physx (upgraded PhysX against the frozen baseline; Steel against upgraded PhysX), debug-instability.
> **Passes when:** every difference is within the declared tolerance or has a named cause that is fixed or accepted in writing.

**Inputs.** A reference trace and a candidate trace with the same seed, task and length (record-baseline-trace);
declared tolerances per quantity. **Tools.** A trace diff by semantic name; the application's backend switch.

1. Align by body and joint name. Compare reset state, first observation and action, policy input order, then every
   step: observations, actions, targets, torques, state, contacts, rewards, terminations.
2. Find the first step and the first quantity that leaves tolerance; that is where to look, not the final outcome.
3. Audit the usual causes, in this order:
   - **the stack, not the solver:** reset timing, quaternion conventions, asset import, stale observation history;
   - **options:** every solver option the application passes reaches the solver (a wrapper that filters keyword
     arguments can drop one); Steel's defaults where PhysX differs: joint limits and joint velocity limits are off
     unless enabled (`enable_joint_limits`, `enable_joint_velocity_limits`), restitution is on
     (`enable_restitution`);
   - **actuation:** gains applied once, the same drive model, torques and effort limits;
   - **forces:** external wrenches and targets re-applied for every internal substep;
   - **materials:** how friction and restitution combine, and the values the solver reads;
   - **contacts:** which pairs collide (collision filters; collision APIs on instanced or inherited prims), the
     realized collision shapes (a hull substituted for a mesh), predictive or speculative contact settings;
   - **capacity:** row watermarks and `check_constraint_capacity()` (size-capacities).
4. When a small difference would change the conclusion, repeat the full matched run within each backend first.
   Fixed seeds do not make GPU rollouts bitwise identical, and deterministic contact ordering is not whole-solver
   determinism. Keep world layout and batch size: a one-world rerun is a different experiment.

**Outputs.** A table: quantity, first step out of tolerance, size, cause, fix or acceptance.

**Fails when** a difference has no cause. For upgraded PhysX against the baseline, that stops the port: no Steel
result can be interpreted until the stack matches.
