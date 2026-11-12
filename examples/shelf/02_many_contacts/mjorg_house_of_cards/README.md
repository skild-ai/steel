# House of cards (mujoco.org)

Steel replica of "House of Cards" from the MuJoCo 4.0 tour on mujoco.org. 26 playing cards, 0.7 mm thick,
make a four-storey house that stands on friction alone.

- Source: `google-deepmind/mujoco` `model/cards/house_of_cards.xml` and its `assets/`, Apache-2.0 (text in
  `LICENSE`). `scene.xml` is that file compiled with MuJoCo 3.17 and re-emitted flat; the card textures
  are downscaled to 256 x 256.
- Check: after 20 s no card has moved more than 5 mm, and no contact rows were dropped.

Known gaps:

- Steel needs Nesterov momentum with contact warm start (serial solve, immediate response) at 2 ms steps.
  With the colored solve and propagation response the apex pair slides apart within 2 s at every setting
  tried.
- MuJoCo's contacts are soft and over-damped (`solref` 0.005 3), with an elliptic cone, `impratio` 10 and
  two noslip iterations; Steel's contacts are hard. Air density and viscosity are not modelled.
- The tour lets the user drag a card; the shelf scene runs untouched.
- Rendering uses the shelf viewer, not MuJoCo's Filament renderer.
