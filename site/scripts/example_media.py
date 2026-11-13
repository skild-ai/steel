# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0
"""Make an example's poster, gallery preview and page clip from a recording of its Steel preset.

usage: python scripts/example_media.py <name> <recording.mp4> --preset steel [--speed 1] [--start S] [--poster T]

Writes public/examples/<name>/{poster.jpg,preview.mp4,clip.mp4} and their source in src/data/example-media.json.
The preview is the 5 s with the most motion unless --start is given; the clip is the first 10 s. Record with
`shelf play <name>` and Record (sim time); --speed is sim seconds per recording second.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parent.parent
WIDTH, HEIGHT, FPS = 640, 360, 30
POSTER_BYTES, PREVIEW_BYTES, CLIP_BYTES = 60_000, 300_000, 600_000
PREVIEW_SECONDS, CLIP_SECONDS = 5.0, 10.0
FFMPEG = "ffmpeg"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("name")
    p.add_argument("recording", type=Path)
    p.add_argument("--preset", required=True, help="the preset the recording shows")
    p.add_argument("--speed", type=float, default=1.0, help="sim seconds per second of the recording")
    p.add_argument("--start", type=float, help="preview start in seconds of the recording (default: most motion)")
    p.add_argument("--poster", type=float, help="poster time in seconds of the recording (default: mid-preview)")
    args = p.parse_args()

    duration = float(
        run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(args.recording)])
    )
    seconds = min(PREVIEW_SECONDS, duration)
    start = busiest_window(args.recording, seconds, duration) if args.start is None else args.start
    poster_t = start + seconds / 2 if args.poster is None else args.poster
    clip_seconds = min(CLIP_SECONDS, duration)
    out = SITE / "public" / "examples" / args.name
    out.mkdir(parents=True, exist_ok=True)
    preview, preview_crf = encode_video(args.recording, start, seconds, out / "preview.mp4", PREVIEW_BYTES)
    clip, clip_crf = encode_video(args.recording, 0.0, clip_seconds, out / "clip.mp4", CLIP_BYTES)
    poster, quality = encode_poster(args.recording, poster_t, out / "poster.jpg")

    manifest_path = SITE / "src" / "data" / "example-media.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["examples"][args.name] = {
        "preset": args.preset, "speed": args.speed, "clip_seconds": round(clip_seconds, 2),
        "preview_start": round(start, 2), "preview_seconds": round(seconds, 2), "poster": round(poster_t, 2),
    }  # fmt: skip
    manifest["examples"] = dict(sorted(manifest["examples"].items()))
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        f"{args.name}: preview {start:.1f}-{start + seconds:.1f} s crf {preview_crf} {preview / 1e3:.0f} KB,"
        f" clip {clip_seconds:.1f} s crf {clip_crf} {clip / 1e3:.0f} KB, poster {poster_t:.1f} s q {quality} {poster / 1e3:.0f} KB"
    )


def busiest_window(recording: Path, seconds: float, duration: float) -> float:
    """Start of the window with the largest summed frame-to-frame change, at 10 frames per second."""
    w, h, rate = 96, 54, 10
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(recording), "-vf", f"fps={rate},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True, stdin=subprocess.DEVNULL,
    ).stdout  # fmt: skip
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    motion = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    n = max(1, int(seconds * rate))
    if len(motion) <= n:
        return 0.0
    sums = np.convolve(motion, np.ones(n), "valid")
    return min(float(np.argmax(sums)) / rate, max(0.0, duration - seconds))


def encode_video(recording: Path, start: float, seconds: float, out: Path, max_bytes: int) -> tuple[int, int]:
    """H.264 yuv420p with faststart, muted; the lowest CRF from 20 up that fits max_bytes."""
    for crf in range(20, 44, 2):
        run([
            FFMPEG, "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(recording), "-an",
            "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}:flags=lanczos", "-c:v", "libx264", "-preset", "veryslow",
            "-crf", str(crf), "-profile:v", "high", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
        ])  # fmt: skip
        if out.stat().st_size <= max_bytes:
            break
    return out.stat().st_size, crf


def encode_poster(recording: Path, t: float, out: Path) -> tuple[int, int]:
    """One frame as JPEG at the best quality that fits POSTER_BYTES."""
    for quality in range(2, 16):
        run([
            FFMPEG, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(recording), "-frames:v", "1",
            "-vf", f"scale={WIDTH}:{HEIGHT}:flags=lanczos", "-q:v", str(quality), str(out),
        ])  # fmt: skip
        if out.stat().st_size <= POSTER_BYTES:
            break
    return out.stat().st_size, quality


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout.strip()


if __name__ == "__main__":
    main()
