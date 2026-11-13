| group | scene | naive | advisor | tuned preset | advisor / tuned frame time | trials to healthy | trials | tune wall (min) | fixes applied |
|---|---|---|---|---|---|---|---|---|---|
| held-out (final) | bolt_sweep | failed - | ok 6.25 | ok 9.32 | 0.67x | 2 | 9 | 6.8 | missing_iterations, grasp_anchors |
| held-out (final) | bounce | ok 1.08 | ok 0.83 | ok 1.24 | 0.66x | 1 | 5 | 0.7 | - |
| held-out (final) | loop_hoberman_sphere | unstable 41.61 | ok 157.48 | ok 106.70 | 1.48x | 4 | 9 | 14.0 | articulation_capacity, loop_gap |
| held-out (final) | loop_parallel_gripper | unstable 1.16 | ok 1.14 | ok 1.14 | 1.00x | 2 | 10 | 1.3 | escalate_substeps |
| held-out (final) | rods | ok 2.36 | ok 0.44 | ok 0.64 | 0.68x | 1 | 5 | 0.9 | - |
| held-out (final) | shapes_in_box | ok 3.06 | ok 0.82 | ok 1.23 | 0.67x | 1 | 5 | 0.8 | - |
| held-out | beer_pong | ok 1.94 | ok 1.02 | ok 3.18 | 0.32x | 2 | 7 | 1.1 | enable_joint_limits |
| held-out | crane_wrecking_ball | failed - | ok 9.25 | ok 10.98 | 0.84x | 2 | 7 | 3.6 | missing_iterations |
| held-out | mjorg_falling_chain | ok 1.15 | ok 0.45 | ok 0.97 | 0.47x | 1 | 6 | 1.0 | - |
| held-out | nut_bolt_sdf | ok 9.98 | suspect (visual) 1.92 | ok 5.47 | 0.35x | 1 | 7 | 2.3 | - |
| dev (run3) | air_hockey | ok 1.60 | ok 0.40 | ok 2.62 | 0.15x | 2 | 7 | 1.2 | enable_joint_limits |
| dev (run3) | anymal_d_many_worlds | unstable 1.98 | ok 2.82 | ok 3.97 | 0.71x | 2 | 7 | 1.3 | articulation_capacity, enable_joint_limits |
| dev (run3) | brick_pour | unstable 15.85 | ok 9.27 | suspect 17.06 | 0.54x | 2 | 9 | 3.9 | articulation_capacity, free_body_capacity, enable_joint_limits, grasp_iterations |
| dev (run3) | mjorg_luggage_carousel | ok 4.83 | ok 3.47 | ok 3.79 | 0.91x | 1 | 7 | 2.6 | - |
| dev (run3) | robot_ur10 | ok 0.54 | ok 0.28 | ok 0.28 | 1.02x | 2 | 8 | 1.2 | enable_joint_limits |
| dev (run3) | spinning_top | ok 2.03 | ok 0.40 | ok 2.68 | 0.15x | 2 | 6 | 1.0 | enable_joint_limits |
| dev (run2) | balance_board | unstable 2.78 | ok 1.89 | ok 5.24 | 0.36x | 2 | 8 | 1.3 | articulation_capacity, enable_joint_limits, grasp_iterations |
| dev (run2) | franka_plate_rack | failed - | ok 4.18 | ok 5.42 | 0.77x | 2 | 7 | 2.5 | missing_iterations |
| dev (run2) | jenga | ok 13.76 | ok 3.39 | ok 5.26 | 0.64x | 2 | 9 | 2.1 | enable_joint_limits |
| dev (run2) | mjorg_gothic_arch | unstable 14.59 | ok 59.45 | ok 42.25 | 1.41x | 5 | 10 | 12.8 | free_body_capacity, escalate_substeps |
| dev (run2) | mjorg_sleeping_dominos | unstable 14.34 | ok 2.27 | ok 3.44 | 0.66x | 2 | 8 | 3.2 | free_body_capacity |
| dev (run2) | robot_g1 | unstable 2.56 | ok 0.72 | ok 0.53 | 1.34x | 2 | 7 | 1.5 | articulation_capacity, enable_joint_limits |
| dev (run1) | basic_urdf | unstable 2.32 | ok 0.49 | ok 0.50 | 0.98x | 2 | 6 | 1.1 | articulation_capacity |
| dev (run1) | domino_spiral | ok 10.65 | ok 1.76 | ok 4.08 | 0.43x | 1 | 7 | 1.3 | - |
| dev (run1) | drawer_fork | unstable 3.09 | suspect 2.13 | ok 4.55 | 0.47x | 2 | 8 | 1.5 | articulation_capacity, enable_joint_limits |
| dev (run1) | loop_fourbar | unstable 1.73 | suspect (visual) 1.30 | ok 3.18 | 0.41x | 2 | 9 | 1.4 | articulation_capacity, enable_joint_limits |
| dev (run1) | mjorg_house_of_cards | unstable 11.18 | unstable 159.54 | suspect 71.02 | 2.25x | none | 6 | 15.0 | free_body_capacity, escalate_substeps, escalate_iterations |
| dev (run1) | newton_cradle | ok 0.72 | ok 0.27 | ok 1.37 | 0.20x | 2 | 8 | 1.1 | enable_joint_limits |
| dev (run1) | robot_allegro_hand | unstable 2.78 | ok 1.44 | ok 1.22 | 1.18x | 2 | 9 | 4.4 | articulation_capacity, enable_joint_limits, grasp_iterations |
| dev (run1) | robot_anymal_d | unstable 1.72 | ok 0.37 | ok 0.37 | 0.99x | 2 | 7 | 1.2 | articulation_capacity, enable_joint_limits |
| dev (run1) | toy_train | unstable 2.27 | ok 0.51 | ok 2.88 | 0.18x | 2 | 7 | 1.1 | articulation_capacity, enable_joint_limits |
| rule source | linkages | unstable 1.25 | ok 0.22 | ok 1.26 | 0.17x | 2 | 6 | 0.9 | articulation_capacity |
| rule source | overlap | unstable 9.23 | ok 2.81 | ok 5.77 | 0.49x | 2 | 8 | 1.7 | free_body_capacity |
| rule source | ratio_stack | unstable 2.40 | ok 3.36 | ok 5.93 | 0.57x | 3 | 9 | 1.7 | mass_ratio_stack |
| rule source | torsion | failed - | ok 2.88 | ok 4.28 | 0.67x | 3 | 9 | 1.4 | construction_contact_torsion_device, articulation_capacity, enable_joint_limits |

- **held-out (final)** (6 scenes): ok naive 3, advisor 6, tuned 6; ok-or-suspect naive 3, advisor 6, tuned 6. Advisor/tuned frame time over healthy advisor arms: geomean 0.82x, median 0.68x (6 scenes). Trials to healthy: median 1.5, max 4.
- **held-out** (4 scenes): ok naive 3, advisor 3, tuned 4; ok-or-suspect naive 3, advisor 4, tuned 4. Advisor/tuned frame time over healthy advisor arms: geomean 0.46x, median 0.41x (4 scenes). Trials to healthy: median 1.5, max 2.
- **dev (run3)** (6 scenes): ok naive 4, advisor 6, tuned 5; ok-or-suspect naive 4, advisor 6, tuned 6. Advisor/tuned frame time over healthy advisor arms: geomean 0.45x, median 0.63x (6 scenes). Trials to healthy: median 2.0, max 2.
- **dev (run2)** (6 scenes): ok naive 1, advisor 6, tuned 6; ok-or-suspect naive 1, advisor 6, tuned 6. Advisor/tuned frame time over healthy advisor arms: geomean 0.78x, median 0.72x (6 scenes). Trials to healthy: median 2.0, max 5.
- **dev (run1)** (9 scenes): ok naive 2, advisor 6, tuned 8; ok-or-suspect naive 2, advisor 8, tuned 9. Advisor/tuned frame time over healthy advisor arms: geomean 0.49x, median 0.45x (8 scenes). Trials to healthy: median 2.0, max 2.
- **rule source** (4 scenes): ok naive 0, advisor 4, tuned 4; ok-or-suspect naive 0, advisor 4, tuned 4. Advisor/tuned frame time over healthy advisor arms: geomean 0.42x, median 0.53x (4 scenes). Trials to healthy: median 2.5, max 3.
- **all** (35 scenes): ok naive 13, advisor 31, tuned 33; ok-or-suspect naive 13, advisor 34, tuned 35. Advisor/tuned frame time over healthy advisor arms: geomean 0.56x, median 0.66x (34 scenes). Trials to healthy: median 2.0, max 5.
