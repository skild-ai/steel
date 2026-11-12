# Luggage carousel (mujoco.org)

Steel replica of "Surface Velocity: Luggage Carousel" from the MuJoCo 4.0 tour on mujoco.org. Three conveyor
belts carry six bags down a cascade onto a baggage-claim carousel, which carries them around. The belts,
rollers and ring are static; only their surfaces move, through Steel's mesh surface velocity.

- Source: `google-deepmind/mujoco` `model/surfacevel/carousel.xml`, Apache-2.0 (text in `LICENSE`).
  `scene.xml` is that file compiled with MuJoCo 3.17 and re-emitted flat (its `attach` and `replicate`
  blocks expanded), without the 4.0 `surfacevel` attributes. `example.py` applies the same velocities:
  it turns each belt box and roller cylinder into a mesh and gives every vertex the geom's
  `linear + angular x r`.
- Check: after 40 s at least four of the six bags are on the carousel and have ridden it at least a quarter
  turn, and no contact rows were dropped. The bags collide as convex hulls, as MuJoCo collides mesh geoms.

Known gaps:

- Newton's surface velocity is per mesh vertex, interpolated over each triangle, so the rollers are
  32-sided meshes rather than true cylinders.
- MuJoCo collides with the ring's convex hull; Newton collides with the ring's triangles, as it must for
  surface velocity.
- Bags sometimes tip over a rail at a belt transfer onto the floor: in 40 runs, 13 lost one or two bags.
  This is accepted; the check asks for four of the six bags on the carousel for that reason.
  In about one run in twenty, two bags wedge side by side between the island and the rail and the bags
  behind them pile up.
- The ring's friction is the average of its 0.1 and the bags' 1; MuJoCo's `priority` gives the ring's 0.1.
  Steel's `friction_combine = "min"` reproduces that, but jammed more often.
- Rendering uses the shelf viewer, not MuJoCo's Filament renderer.
