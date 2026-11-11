"""The three ways to run an example: play, throughput and tune.

All three run the example's own ``__main__`` block, so its argument parsing and scene
setup are untouched. Throughput and tune replace ``newton.examples.run`` -- the loop the
example hands itself to at the end -- with their own loop.

* play       -- Newton's normal loop: the viewer, mouse picking, pause, reset.
* throughput -- headless; times ``Example.step`` over many frames at the world count
                you pass (``--world-count``) and reports step time, throughput and peak
                GPU memory.
* tune       -- the viewer plus a panel listing the preset's settings. Edit values, press
                Apply to make them take effect (live where the solver reads them, else by
                swapping the solver or rebuilding the scene; see common/live.py), and Save as
                to write the tuned preset to a new TOML file you name, next to presets.toml
                (load it with --presets).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import gc
import json
import runpy
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import defaults, live, looks, panel, presets


def run_example(path: Path, argv: list[str], loop=None, look: str | None = None, host=None) -> None:
    """Execute ``path`` as ``__main__`` with ``argv``; ``loop`` replaces ``newton.examples.run``.

    ``look`` names a look from looks.toml to attach to the viewer before the loop starts.
    ``host`` (play mode) is a :class:`PlayHost` that takes over the side bar while Newton's own
    loop runs the example.
    """
    import newton.examples

    sys.argv = [str(path), *argv]
    sys.path.insert(0, str(path.parent))
    import newton.viewer

    original = newton.examples.run
    original_viewer = newton.viewer.ViewerGL
    # a loop or host that wants the viewer as soon as it exists (the tuner, the play-mode bar)
    attach = getattr(loop, "attach", None) or getattr(host, "attach", None)
    owner = loop if hasattr(loop, "next_example") else host  # whoever can ask for the next example
    if host is not None and loop is None:
        inner_run = original

        def loop(example, args):
            host.bind(example)
            # Newton's Reset builds the next example inside its loop; tell the bar about each one
            cls = type(example)
            init = cls.__init__

            def init_and_bind(self, *a, **k):
                init(self, *a, **k)
                host.bind(self)

            cls.__init__ = init_and_bind
            # an example picked in the bar ends Newton's loop without closing the window it runs next in
            viewer = example.viewer
            running, close = viewer.is_running, viewer.close
            viewer.is_running = lambda: host.next_example is None and running()
            viewer.close = lambda: None if host.next_example is not None else close()
            try:
                return inner_run(example, args)
            finally:
                del viewer.is_running, viewer.close
                if host.next_example is not None:
                    host.example = None  # the next example builds in this window

    if look is not None:
        looks.load(look)  # fail before the example builds if the name is wrong
        inner = loop or original

        def loop(example, args):
            if looks.current(example.viewer) is None:  # not attached in the viewer's constructor below
                looks.attach(example.viewer, look)
            return inner(example, args)

    kept = _window["viewer"]
    if kept is not None:
        # an example picked in the bar: newton.examples.init gets the open window back instead of a new one
        _clear_scene(kept)

        # still a class, so the examples' `isinstance(viewer, newton.viewer.ViewerGL)` checks keep working
        class _AnyViewerGL(type(original_viewer)):
            def __instancecheck__(cls, obj):
                return isinstance(obj, original_viewer)

        class _ReusedViewerGL(original_viewer, metaclass=_AnyViewerGL):
            def __new__(cls, *a, **k):
                # not an instance of this class, so Python skips __init__ and the window is reused as is
                if attach is not None:
                    attach(kept)
                return kept

        newton.viewer.ViewerGL = _ReusedViewerGL
    elif attach is not None:
        # newton.examples.init builds the viewer and draws the loading splash before it returns, so
        # the side bar has to be taken over in the viewer's constructor to never show Newton's panel
        class _AttachedViewerGL(original_viewer):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                attach(self)
                if look is not None:
                    looks.attach(self, look)

        newton.viewer.ViewerGL = _AttachedViewerGL
    if loop is not None:
        newton.examples.run = loop
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        newton.examples.run = original
        newton.viewer.ViewerGL = original_viewer
        with contextlib.suppress(ValueError):
            sys.path.remove(str(path.parent))
    _window["viewer"] = owner.viewer if getattr(owner, "next_example", None) is not None else None


# The window an example picked in the side bar runs in: set while one example hands over to the next.
_window: dict[str, Any] = {"viewer": None}


def _clear_scene(viewer) -> None:
    """Drop the last example's models, GUI and GPU memory from ``viewer`` before the next one builds."""
    if hasattr(viewer, "clear_all_layers"):
        viewer.clear_all_layers()
    else:
        viewer.clear_model()
    viewer.picking_enabled = True  # each example starts with picking on and may turn it off
    live.reset_record()
    _forget_main_kernels()
    gc.collect()  # frees the last example, its CUDA graph and the kernel objects just dropped
    if hasattr(viewer, "_paused"):
        viewer._paused = False
    for _ in range(2):  # paint the bar's "loading" before the next example blocks the window to build
        viewer.begin_frame(0.0)
        viewer.end_frame()


def _forget_main_kernels() -> None:
    """Drop the kernels the last example defined at module level from Warp's ``__main__`` module.

    Every example runs as ``__main__``, so Warp files the ``@wp.kernel``, ``@wp.func`` and
    ``@wp.struct`` definitions of all of them in one module. Warp only resets that module when
    ``sys.modules["__main__"]`` changes loader, and runpy gives every example file the same loader,
    None, so left alone the module grows with each example visited: the first visit to an example
    with kernels recompiles everything visited so far, and none of those builds matches the
    per-example ones ``shelf check`` already cached. This does what Warp does itself on a loader
    change (``_get_module`` in warp/_src/context.py): unload the executable and empty the
    registries, so the next example compiles, or loads from the cache, only its own kernels.

    Warp hashes every kernel object still alive, and each sits in a reference cycle with its file's
    globals (the file names the kernel; the kernel's function holds the globals), so the caller
    collects garbage right after this; without that the next build would still carry the last
    example's kernels. Nothing alive launches a dropped kernel: the last example is unreachable by
    now and goes in that same collection.
    """
    import warp as wp

    module = wp.get_module("__main__")
    module.unload()
    for registry in ("kernels", "functions", "structs"):
        getattr(module, registry, {}).clear()


def _sync() -> None:
    import warp as wp

    wp.synchronize_device()


RECORDINGS = Path(__file__).resolve().parents[1] / "recordings"


@dataclasses.dataclass
class PendingRecording:
    """A finished recording waiting for the bar's save-or-discard choice."""

    path: Path  # the temporary mp4 (or, without ffmpeg, the folder of PNG frames)
    frames: int  # video frames
    fps: float
    name: str  # suggested file name
    folder: Path  # suggested folder
    mode: str = "sim"  # "sim": frames are simulation frames; "real": the video follows the wall clock

    @property
    def seconds(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    def size_mb(self) -> float:
        if self.path.is_dir():
            return sum(f.stat().st_size for f in self.path.iterdir()) / 2**20
        return self.path.stat().st_size / 2**20 if self.path.exists() else 0.0


class Recording:
    """Viewport recording shared by the tuner and play mode: Record, Stop, then save or discard.

    Frames go to a temporary file under ``recordings/`` while recording; after Stop the bar
    shows a dialog and :meth:`save_recording` moves the file where the user chose, or
    :meth:`discard_recording` deletes it.
    """

    example: Any
    example_name: str
    message: str

    def _init_recording(self) -> None:
        self.recorder: Recorder | None = None
        self.pending_recording: PendingRecording | None = None
        self.recording_stamp = ""
        self.record_mode = "sim"  # "sim" or "real"; the bar's "record as" switch

    def start_recording(self) -> None:
        mode = getattr(self, "record_mode", "sim")
        fps = REAL_TIME_FPS if mode == "real" else float(getattr(self.example, "fps", 0) or 60.0)
        self.recording_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        RECORDINGS.mkdir(parents=True, exist_ok=True)
        self.recorder = Recorder(RECORDINGS / f".pending_{self.recording_stamp}.mp4", fps, mode)
        self.message = "recording"

    def stop_recording(self) -> None:
        recorder, self.recorder = self.recorder, None
        if recorder is None:
            return
        path = Path(recorder.finish())
        if recorder.emitted == 0:
            self.discard_path(path)
            self.message = "nothing was recorded"
            return
        preset = getattr(self, "preset_name", None)
        name = f"{self.example_name}{'_' + preset if preset else ''}_{self.recording_stamp}{path.suffix}"
        self.pending_recording = PendingRecording(path, recorder.emitted, recorder.fps, name, RECORDINGS, recorder.mode)
        self.message = ""

    def save_recording(self, folder: str | Path, name: str) -> None:
        pending = self.pending_recording
        if pending is None:
            return
        folder = Path(folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        name = name.strip() or pending.name
        if pending.path.suffix and not name.endswith(pending.path.suffix):
            name += pending.path.suffix
        target = folder / name
        n = 1
        while target.exists():  # never overwrite silently
            target = folder / f"{Path(name).stem}-{n}{Path(name).suffix}"
            n += 1
        shutil.move(str(pending.path), str(target))
        self.message = f"saved {pending.frames} frames ({pending.seconds:.1f} s) to {target}"
        self.pending_recording = None

    def discard_recording(self) -> None:
        pending = self.pending_recording
        if pending is None:
            return
        self.discard_path(pending.path)
        self.message = "recording discarded"
        self.pending_recording = None

    @staticmethod
    def discard_path(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _bar_for(host, viewer, mode: str) -> panel.Panel:
    """The side bar ``host`` draws in: a new one on a new viewer; on the window an example was
    picked in, the bar already there, handed over to ``host``."""
    bar = getattr(viewer, "_shelf_bar", None)
    if bar is None:
        bar = panel.Panel(host, mode=mode)
        _take_over_bar(viewer, bar)
        viewer._shelf_bar = bar
    else:
        bar.adopt(host, mode)
    return bar


def _take_over_bar(viewer, bar: panel.Panel) -> None:
    """Draw the shelf's side bar instead of Newton's panel, and its stats card instead of Newton's
    FPS overlay, for the life of ``viewer``.

    The bar is a ``free`` UI callback, and Newton's GL viewer drops every ``free`` and ``side``
    callback when it clears the scene (``clear_model``, on Reset and on a tuner rebuild), so the
    bar puts itself back after each clear. Newton's loading splash is turned off: the bar says
    what is happening, and the look's sky stays as it will be.
    """
    draw = bar.draw
    viewer.register_ui_callback(draw, position="free")
    gui = getattr(viewer, "gui", None)
    if gui is None:
        return
    if hasattr(gui, "_render_left_panel"):
        gui._render_left_panel = lambda: None  # the bar's Scene section draws Newton's controls
    if hasattr(gui, "_render_stats_overlay"):
        gui._render_stats_overlay = lambda: bar.draw_stats(gui)
    clear = getattr(gui, "clear_example_callbacks", None)
    if clear is not None:

        def clear_but_keep_bar():
            clear()
            if draw not in gui._ui_callbacks["free"]:
                gui.register_ui_callback(draw, position="free")

        gui.clear_example_callbacks = clear_but_keep_bar
    if hasattr(viewer, "show_loading_splash"):
        viewer.show_loading_splash = lambda *a, **k: None
    window = getattr(getattr(viewer, "renderer", None), "window", None)
    if window is not None and hasattr(window, "push_handlers"):
        from pyglet.window import key as pyglet_key

        def on_key_press(symbol, modifiers):
            # Escape closes the browser over a running example; pyglet's default would close the window
            browser = bar.gallery
            if symbol == pyglet_key.ESCAPE and browser.open and getattr(browser.host, "path", None) is not None:
                browser.hide()
                return True
            return None

        window.push_handlers(on_key_press=on_key_press)


class PlayHost(Recording):
    """What the side bar needs in play mode, while Newton's own loop runs the example.

    Pause, step and reset go through the viewer (Newton's loop reads ``_paused``,
    ``_step_requested`` and the reset callback its example browser installs); recording hooks
    ``viewer.end_frame``.
    """

    def __init__(self, path: Path | None):
        self.path = path  # None: the window shows only the example browser (`shelf play` with no example)
        self.example_name = path.parent.name if path is not None else ""
        self.viewer = None
        self.example = None
        self.panel = None
        self.message = ""
        self.next_example: str | None = None  # picked in the bar: the loop ends and this one runs next
        self.next_solver: str | None = None  # the preset it opens with (a chip in the browser); None: its default
        self._init_recording()

    def attach(self, viewer) -> None:
        """Take over the side bar as soon as the viewer exists (see :meth:`Tuner.attach`)."""
        if self.viewer is viewer or not hasattr(viewer, "register_ui_callback"):
            return
        self.viewer = viewer
        self.panel = _bar_for(self, viewer, "play")
        if getattr(viewer, "_shelf_records", False):
            return  # a window an example was picked in: the hook below records for whichever host has the bar
        end_frame = viewer.end_frame

        def end_frame_and_record(*a, **k):
            end_frame(*a, **k)
            recorder = getattr(viewer._shelf_bar.host, "recorder", None)
            if recorder is not None:
                recorder.capture(viewer)

        viewer.end_frame = end_frame_and_record
        viewer._shelf_records = True

    def bind(self, example) -> None:
        self.example = example

    def can_reset(self) -> bool:
        return getattr(self.viewer, "_reset_callback", None) is not None

    def reset(self) -> None:
        callback = getattr(self.viewer, "_reset_callback", None)
        if callback is not None:
            self.example = None  # let the old scene's GPU memory go before the new one builds
            callback()


def run_gallery(look: str | None, building: int = 0, total: int = 0) -> PlayHost:
    """``shelf play`` with no example: a window with the look's floor and sky and the example browser
    over it, until an example is picked (the window is then kept for it) or the window is closed.

    ``building`` and ``total`` say how many previews a background build is rendering, for the
    browser's first-visit line. ``SHELF_GALLERY_PICK=example[:preset]`` picks without a click, for
    tests and screenshots.
    """
    import os

    import newton
    import newton.examples
    import newton.viewer

    viewer = newton.viewer.ViewerGL()
    host = PlayHost(None)
    host.attach(viewer)
    if look is not None:
        looks.attach(viewer, look)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    viewer.set_model(builder.finalize())
    host.panel.gallery.show(host, building, total)
    pick = os.environ.get("SHELF_GALLERY_PICK")
    frames = 0
    while viewer.is_running() and host.next_example is None:
        start = time.perf_counter()
        viewer.begin_frame(0.0)
        viewer.end_frame()
        frames += 1
        if pick and frames == 3:
            name, _, preset = pick.partition(":")
            host.panel.gallery.pick(name, preset or None)
        newton.examples._throttle_render_fps(start, 60.0)
    if host.next_example is None:
        viewer.close()
        _window["viewer"] = None
    else:
        _window["viewer"] = viewer
    return host


REAL_TIME_FPS = 60.0


class Recorder:
    """Write the viewer's frames to an mp4 (through ffmpeg) or, without ffmpeg, to PNG files.

    ``mode`` is ``"sim"`` (one video frame per captured frame, played at ``fps``, the example's
    frame rate: the video runs at simulation time) or ``"real"`` (a ``fps`` video whose length
    matches the wall clock: a captured frame is written once per elapsed video slot, so slow
    rendering repeats frames and fast rendering drops them).
    ``target`` is the mp4 to write; without ffmpeg the frames go to a folder of the same name.
    """

    def __init__(self, target: Path, fps: float, mode: str = "sim"):
        self.output = target
        self.folder = target.with_suffix("")
        self.fps = fps
        self.mode = mode
        self.count = 0  # frames captured
        self.emitted = 0  # frames written to the video
        self.next_slot = 0.0  # real time: wall-clock time of the next video frame
        self.proc: subprocess.Popen | None = None
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            self.folder.mkdir(parents=True, exist_ok=True)

    def _repeat(self) -> int:
        """How many video frames this capture stands for."""
        if self.mode != "real":
            return 1
        now = time.perf_counter()
        if self.count == 0:
            self.next_slot = now + 1.0 / self.fps
            return 1
        if now < self.next_slot:
            return 0
        slots = int((now - self.next_slot) * self.fps) + 1
        self.next_slot += slots / self.fps
        return slots

    def capture(self, viewer) -> None:
        repeat = self._repeat()
        frame = viewer.get_frame().numpy() if repeat else None  # height x width x 3, uint8
        self.count += 1
        if repeat == 0:
            return
        if self.ffmpeg is not None:
            if self.proc is None:
                h, w = frame.shape[:2]
                self.output.parent.mkdir(parents=True, exist_ok=True)
                self.proc = subprocess.Popen(
                    [
                        self.ffmpeg,
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{w}x{h}",
                        "-framerate",
                        f"{self.fps:g}",
                        "-i",
                        "-",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-crf",
                        "18",
                        str(self.output),
                    ],
                    stdin=subprocess.PIPE,
                )
            data = frame.tobytes()
            for _ in range(repeat):
                self.proc.stdin.write(data)
        else:
            from PIL import Image

            image = Image.fromarray(frame)
            for i in range(repeat):
                image.save(self.folder / f"frame_{self.emitted + i:05d}.png")
        self.emitted += repeat

    def finish(self) -> str:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            return str(self.output)
        return str(self.folder)


def _mem_mb(high: bool) -> float | None:
    import warp as wp

    device = wp.get_device()
    if not device.is_cuda or not wp.is_mempool_enabled(device):
        return None
    used = wp.get_mempool_used_mem_high(device) if high else wp.get_mempool_used_mem_current(device)
    return used / 2**20


def _process_gpu_mb() -> float | None:
    """This process's total GPU memory (CUDA context and all allocators) from nvidia-smi."""
    import os
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        pid, _, mem = line.partition(",")
        if pid.strip() == str(os.getpid()):
            return float(mem)
    return None


# --- throughput ------------------------------------------------------------------------------------


class Throughput:
    """Replacement for ``newton.examples.run`` that benchmarks ``Example.step``."""

    def __init__(self, example_name: str, frames: int, warmup: int, json_path: str | None):
        self.example_name = example_name
        self.frames = frames
        self.warmup = warmup
        self.json_path = json_path
        self.build_start = time.perf_counter()
        self.ran = False  # stays False when the example never calls newton.examples.run()

    def __call__(self, example, args) -> None:
        import warp as wp

        self.ran = True
        build_s = time.perf_counter() - self.build_start
        # Examples with several models (side-by-side solvers) have no single `model`; count one world.
        model = getattr(example, "model", None)
        worlds = int(getattr(args, "world_count", None) or getattr(model, "world_count", 1) or 1)
        substeps = int(getattr(example, "sim_substeps", 1) or 1)
        preset = getattr(example, "preset", None)

        for _ in range(self.warmup):
            example.step()
        _sync()
        times = []
        for _ in range(self.frames):
            start = time.perf_counter()
            example.step()
            _sync()
            times.append(time.perf_counter() - start)
        total = sum(times)

        result = {
            "example": self.example_name,
            "preset": preset.name if preset else None,
            "device": wp.get_device().name,
            "worlds": worlds,
            "substeps": substeps,
            "frames": self.frames,
            "warmup_frames": self.warmup,
            "build_s": round(build_s, 3),
            "frame_ms_mean": 1e3 * total / len(times),
            "frame_ms_median": 1e3 * statistics.median(times),
            "frame_ms_p95": 1e3 * sorted(times)[int(0.95 * (len(times) - 1))],
            "substep_ms_mean": 1e3 * total / len(times) / substeps,
            "frames_per_s": len(times) / total,
            "world_steps_per_s": worlds * len(times) / total,
            "world_substeps_per_s": worlds * substeps * len(times) / total,
            "warp_mem_peak_mb": _mem_mb(high=True),
            "process_gpu_mem_mb": _process_gpu_mb(),
        }
        self.report(result, has_world_count=hasattr(args, "world_count"))
        if self.json_path:
            with open(self.json_path, "a") as f:
                f.write(json.dumps(result) + "\n")

    @staticmethod
    def report(r: dict[str, Any], has_world_count: bool) -> None:
        warp_mem = "n/a (mempool off)" if r["warp_mem_peak_mb"] is None else f"{r['warp_mem_peak_mb']:.1f} MB"
        proc_mem = "n/a" if r["process_gpu_mem_mb"] is None else f"{r['process_gpu_mem_mb']:.0f} MB"
        print(f"\n{r['example']} [{r['preset']}] on {r['device']}")
        rows = [
            ("worlds", f"{r['worlds']}" + ("" if has_world_count else "  (this example has no --world-count)")),
            ("substeps per frame", f"{r['substeps']}"),
            ("frames timed", f"{r['frames']} after {r['warmup_frames']} warm-up"),
            ("build time", f"{r['build_s']:.2f} s"),
            ("frame step time", f"mean {r['frame_ms_mean']:.3f} ms, median {r['frame_ms_median']:.3f} ms, "
                                f"p95 {r['frame_ms_p95']:.3f} ms"),
            ("substep time", f"mean {r['substep_ms_mean']:.3f} ms"),
            ("throughput", f"{r['frames_per_s']:.1f} frames/s, {r['world_steps_per_s']:,.0f} world-frames/s, "
                           f"{r['world_substeps_per_s']:,.0f} world-substeps/s"),
            ("GPU memory", f"process {proc_mem} (incl. CUDA context); Warp allocations peak {warp_mem}"),
        ]  # fmt: skip
        for label, value in rows:
            print(f"  {label:20s} {value}")


# --- tune ------------------------------------------------------------------------------------------


def _flatten(table: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for key, value in table.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(_flatten(value, dotted + "."))
        else:
            out[dotted] = value
    return out


class Tuner(Recording):
    """Replacement for ``newton.examples.run`` with a live preset-editing panel."""

    def __init__(self, path: Path, overrides: dict[str, Any], presets_file: str | None = None):
        self.path = path
        self.presets_file = presets_file  # the file the preset was loaded from (None: presets.toml)
        self.save_name = ""
        self.overwrite = False
        self.example_name = path.parent.name
        self.overrides = dict(overrides)  # applied (rebuilt) values
        self.edits: dict[str, Any] = {}  # typed but not yet applied
        self.new_key = ""
        self.message = ""
        self.rebuild = False
        self.pending: dict[str, Any] = {}  # changes Apply asked for, taken by the loop
        self.live_applied = False  # a live/model/swap change is in effect since the last build
        self.switch_to: str | None = None
        self.next_example: str | None = None  # picked in the bar: the loop ends and this one runs next
        self.next_solver: str | None = None  # the preset it opens with (a chip in the browser); None: its default
        self.step_times: list[float] = []
        self._init_recording()
        self.example = None  # set by __call__; None while the scene builds or after a failed rebuild
        self.viewer = None
        self.args = None
        self.panel = None
        self.ran = False  # stays False when the example never calls newton.examples.run()
        live.install_recorder()  # the swap tier needs to know how the example built its solver

    # The loop mirrors newton.examples.run, with rebuilds driven by the panel.
    def __call__(self, example, args) -> None:
        import copy

        import newton.examples

        self.ran = True
        viewer = example.viewer
        example_class = type(example)
        self.args = copy.deepcopy(args)
        self.example = example
        if not hasattr(viewer, "register_ui_callback"):
            sys.exit("tune mode needs the OpenGL viewer (--viewer gl)")
        self.attach(viewer)
        if hasattr(viewer, "hide_loading_splash"):
            viewer.hide_loading_splash()  # newton.examples.init shows it; run() normally hides it
        render_fps = getattr(args, "render_fps", None)
        if hasattr(viewer, "set_reset_callback"):
            # The viewer's Reset rebuilds with the applied settings.
            viewer.set_reset_callback(lambda: setattr(self, "rebuild", True))
        self._register_example_gui(viewer)

        while viewer.is_running() and self.next_example is None:
            frame_start = time.perf_counter()
            if self.pending:
                self.apply_pending()
            if self.rebuild:
                self.rebuild = False
                self.example = None
                gc.collect()
                if hasattr(viewer, "clear_all_layers"):
                    viewer.clear_all_layers()
                else:
                    viewer.clear_model()
                presets.select(self.preset_name, self.overrides, self.presets_file)
                live.reset_record()
                why = self.message[len("rebuilding: ") :] if self.message.startswith("rebuilding: ") else ""
                self.live_applied = False
                try:
                    self.example = example_class(viewer, copy.deepcopy(self.args))
                    self.message = f"rebuilt [{self.preset_name}]" + (f" ({why})" if why else "")
                except Exception as e:  # keep the window alive; show the error
                    self.message = f"rebuild failed: {e}"
                    self.example = None
                self.step_times.clear()
                self._register_example_gui(viewer)
            if self.example is None:
                viewer.begin_frame(0.0)
                viewer.end_frame()
                newton.examples._throttle_render_fps(frame_start, render_fps)
                continue
            if viewer.should_step():
                start = time.perf_counter()
                try:
                    self.example.step()
                    _sync()
                except Exception as e:
                    if self.live_applied:
                        self.message = f"rebuilding: step failed after a live change ({e})"
                        self.rebuild = True
                        continue
                    raise
                self.step_times = (self.step_times + [time.perf_counter() - start])[-120:]
            self.example.render()
            if self.recorder is not None and not getattr(viewer, "_shelf_records", False):
                self.recorder.capture(viewer)  # a window from the browser records in its end_frame hook already
            newton.examples._throttle_render_fps(frame_start, render_fps)
        if self.next_example is None:
            viewer.close()
        self.example = None  # the next example builds in this window; let this one's GPU memory go

    def apply_pending(self) -> None:
        """Make the changes Apply asked for take effect with the cheapest tier (see common/live.py).

        Falls back to a full rebuild when a change needs one or the live path fails.
        """
        changes, self.pending = self.pending, {}
        new_overrides = {**self.overrides, **changes}
        if self.example is None:
            self.overrides = new_overrides
            self.rebuild = True
            return
        presets.select(self.preset_name, new_overrides, self.presets_file)
        try:
            preset = presets.load(self.path)
        except presets.PresetError as e:  # a value the preset loader rejects: keep what runs
            presets.select(self.preset_name, self.overrides, self.presets_file)
            self.message = str(e)
            return
        try:
            used = live.apply(self.example, changes, preset)
        except live.NeedsRebuild as e:
            self.overrides = new_overrides
            self.rebuild = True
            self.message = f"rebuilding: {e}"
            return
        self.overrides = new_overrides
        self.live_applied = True
        self.step_times.clear()
        self.message = f"applied ({used}): {', '.join(changes)}"

    def attach(self, viewer) -> None:
        """Take over the viewer's side bar: called as soon as the viewer exists, before the scene builds.

        ``run_example`` calls this from ``newton.examples.init`` so the bar is ours from the first
        frame; ``__call__`` calls it again, which does nothing the second time.
        """
        if self.viewer is viewer or not hasattr(viewer, "register_ui_callback"):
            return
        self.viewer = viewer
        self.panel = _bar_for(self, viewer, "tune")

    def reset(self) -> None:
        """Start the scene over with the applied settings (Newton's Reset)."""
        self.switch_to = self.preset_name
        self.rebuild = True

    def _register_example_gui(self, viewer) -> None:
        if self.example is not None and hasattr(self.example, "gui"):
            viewer.register_ui_callback(lambda ui, ex=self.example: ex.gui(ui), position="side")

    @property
    def preset(self):
        return self.example.preset if self.example is not None and hasattr(self.example, "preset") else None

    @property
    def preset_name(self) -> str:
        if self.switch_to:
            return self.switch_to
        if self.preset:
            return self.preset.name
        return presets.selected() or presets.read(self.path).get("default")

    def current_settings(self) -> dict[str, Any]:
        """Flattened settings of the preset being tuned, with applied overrides."""
        data = presets.read(self.path)
        block = data.get(self.preset_name, {})
        flat = _flatten({k: v for k, v in block.items() if k in (*presets.SECTIONS, "entries")})
        defs = defaults.solver_defaults(self.example, self.preset) if self.example is not None else {}
        for key, value in self.overrides.items():
            if key not in flat and key in defs and value == defs[key].value:
                continue  # a default the tuner set back: it is not part of the preset
            flat[key] = value
        return dict(sorted(flat.items()))

    def save(self, name: str, file_name: str, overwrite: bool = False) -> str:
        """Write preset ``name`` with every applied setting to a new TOML file in the example's folder.

        The file has the same layout as presets.toml (``example``, ``source``, ``default``, ``[check]``
        and one table), so ``--presets <file>`` loads it for play, bench, tune and check. presets.toml
        itself is never written.
        """
        import re

        import tomlkit

        file_name = file_name.strip()
        if not file_name.endswith(".toml"):
            file_name += ".toml"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.toml", file_name):
            return f"invalid file name {file_name!r}: use letters, digits, '_', '-', '.'"
        if file_name == "presets.toml":
            return "presets.toml is the shelf's checked presets; choose another name"
        target = self.path.parent / file_name
        if target.exists() and not overwrite:
            return f"{file_name} already exists; choose another name or tick overwrite"

        source = presets.read(self.path)  # the file the preset was loaded from
        block = source.get(name, {})
        settings: dict[str, Any] = {}
        for dotted, value in self.current_settings().items():
            if value is None:
                continue  # auto: the solver chooses it, which is what leaving it out means (TOML has no null)
            node = settings
            *parents, leaf = dotted.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value

        doc = tomlkit.document()
        doc["example"] = self.example_name
        doc["source"] = source.get("source", "")
        doc["default"] = name
        doc["tuned_from"] = f"{presets.presets_path(self.path).name} [{name}]"
        check = tomlkit.table()
        check["args"] = source.get("check", {}).get("args", [])
        doc["check"] = check
        table = tomlkit.table()
        if "kind" in block:
            table["kind"] = block["kind"]
        table["status"] = "untested"
        table["note"] = (
            f"tuned with `shelf tune` on {datetime.date.today().isoformat()}; check with "
            f"`python -m shelf check {self.example_name} --presets {file_name}`"
        )
        for section in (*presets.SECTIONS, "entries"):
            if section in settings:
                table[section] = _inline(settings[section])
        doc[name] = table
        target.write_text(tomlkit.dumps(doc))
        return (
            f"saved [{name}] to {file_name}; play it with `python -m shelf play {self.example_name} "
            f"--presets {file_name}`"
        )


def _inline(value: Any):
    """Nested dicts as TOML inline tables, so each section stays on one line."""
    import tomlkit

    if isinstance(value, dict):
        table = tomlkit.inline_table()
        for key, item in value.items():
            table[key] = _inline(item)
        return table
    return value
