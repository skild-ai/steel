# Magnetic door (mujoco.org)

Steel replica of "Magnetic Adhesion" from the MuJoCo 4.0 tour on mujoco.org. A fridge door hangs on a
sprung hinge, and a magnet strip on its edge faces a strip on the cabinet frame. The ajar door snaps shut
and stays latched until it is pulled harder than the magnets hold.

- Source: `google-deepmind/mujoco` `model/adhesion/fridge_door.xml`, Apache-2.0 (text in `LICENSE`).
  `scene.xml` is that file as it is upstream. Newton's MJCF importer reads the strips' `adhesion` (1.5 N
  per contact point) into `steel:adhesion` and their `gap` (3 mm) into the shape gap, and Steel pulls on
  each contact of the pair with the sum, 3 N, while it is within the gap. Both Steel and MuJoCo 4.0 snap
  the door shut from 0.0135 rad and leave it open at 0.015 rad, a band of about 6 mm at the strips.
- Script: the door starts at the XML's `ajar` keyframe. The tour's mouse drags become two tugs on the
  handle, applied as hinge torque at 0.46 m: 4 N for 0.3 s at 1 s and 25 N for 0.15 s at 2 s.
- Check, over 8 s: the door is shut (within 0.002 rad) from 0.5 s, stays shut through the light tug, opens
  past 20 degrees on the hard tug, and is latched again for the last 0.5 s; no contact rows were dropped.
  In Steel it shuts within the first frame, opens to its 135 degree stop and latches again at 6.4 s.
  MuJoCo 4.0 on the same file and script passes the same phases and latches again at 4.8 s.

Known gaps:

- The preset's 5 mm rigid gap keeps other shapes from making contacts with the magnets. With Newton's
  default 0.1 m the magnets also pull on the cabinet walls and shelf, and the hard tug cannot open the door.
- Pull-off at the handle is 12.5-14 N in Steel and 11.5-12.5 N in MuJoCo 4.0. Both hold the magnet pair
  with four contact points.
- Inside the gap MuJoCo turns friction off; Steel bounds it by the solved impulse instead.
- The hinge limit is hard in Steel (the door stops at 135 degrees) and soft in MuJoCo (138 degrees), and the
  door takes about 1.5 s longer to swing back in Steel.
- Rendering uses the shelf viewer, not MuJoCo's Filament renderer.
