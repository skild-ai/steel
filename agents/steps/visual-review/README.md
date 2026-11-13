# Visual review

> **Does:** looks at frames across each episode and records a grade and a note per scene the metrics may have misjudged.
> **Use in:** validate-a-solver-change, tune-a-scene, add-an-example, debug-instability.
> **Passes when:** every rendered scene's contact sheet has been looked at and `visual_review.json` covers every scene you have an opinion on.

**Inputs.** A collected run dir from render-and-grade. **Tools.** `ffmpeg`, `collect.py`.

1. Open every `media/SCENE_sheet.jpg` (12 frames across the episode). Look for jitter, explosions,
   interpenetration, launched or floating bodies, and bodies vanishing into geometry.
2. When a sheet looks off, pull single frames:
   ```bash
   ffmpeg -ss 7.5 -i "$RUN/out/SCENE/clip_raw.mp4" -frames:v 1 frame.jpg
   ```
3. Before calling a body lost, find it in the poses: a fixed camera cannot see into a bin behind a deck, and the
   example's own check may count it. Compare doubtful metrics with another solver on the same scene.
4. Write `$RUN/visual_review.json`:
   ```json
   {"crane_wrecking_ball": {"grade": "ok", "note": "the ball knocks the building down as scripted; nothing launches or sinks"}}
   ```
   then run `collect.py` and `build_gallery.py` again.

**Rules.** A visual note can raise a grade; it never hides an automatic `unstable` or `failed`. Say what you saw, at
what time, in which part of the frame.

**Outputs.** `visual_review.json` and a regenerated `results.json` and gallery.
