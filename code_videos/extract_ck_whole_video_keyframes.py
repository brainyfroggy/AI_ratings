r"""Extract CK whole-video key frames into one folder per video.

This uses the same frame-selection path as ck_whole_video_script:

- frame count = round(duration_seconds * 1 fps), clamped to 3..20
- full-video context window [0, video end]
- CK default sampling mode "auto": uniform for clips up to 10 s, histogram
  key-moment selection for longer clips

By default, frames are written to:
    N:\Experimental_Data\yujunchen\projects\AI_ratings\ck_whole_video_keyframes

Each video gets a subfolder named by video stem, for example:
    ck_whole_video_keyframes\0001\frame_001_000000ms.jpg
    ck_whole_video_keyframes\0001\metadata.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CK_SCRIPT_ROOT = PROJECT_ROOT  # src/ lives at the repo root
DEFAULT_VIDEO_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\original_ckvideo_data\ckvideo"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "ck_whole_video_keyframes"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CK whole-video frames with the ck_whole_video_script algorithm."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sampling",
        choices=["auto", "uniform", "key_moments"],
        default="auto",
        help="Frame selection mode. Default auto matches ck_whole_video_script.",
    )
    parser.add_argument(
        "--key-moment-threshold-sec",
        type=float,
        default=10.0,
        help="Auto mode uses key-moment selection above this duration.",
    )
    parser.add_argument("--frame-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--videos", nargs="*", default=None, help="Exact stems or filenames.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip videos with metadata.json.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing per-video output folder before extraction.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def id_to_sort_key(value: str) -> tuple[float, int, str]:
    stem = Path(value).stem
    try:
        return (float(stem), 0, value)
    except ValueError:
        return (float("inf"), 1, value)


def list_videos(video_dir: Path, requested: list[str] | None) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    videos = [
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]
    videos.sort(key=lambda path: id_to_sort_key(path.name))

    if requested:
        wanted = {item.lower() for item in requested}
        videos = [
            path
            for path in videos
            if path.name.lower() in wanted or path.stem.lower() in wanted
        ]

    return videos


def video_duration_ms(path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            return 5000.0, fps, frame_count
        # Frames are numbered 0..frame_count-1. Use the timestamp of the last
        # real frame so the extractor requests a readable endpoint.
        return ((frame_count - 1) / fps) * 1000.0, fps, frame_count
    finally:
        cap.release()


def load_ck_components() -> tuple[Any, Any]:
    sys.path.insert(0, str(CK_SCRIPT_ROOT))
    from src.frame_extractor import FrameExtractor
    from src.rating_strategy import STRATEGY_CK

    return FrameExtractor, STRATEGY_CK


def extract_one_video(
    video_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    frame_extractor_cls: Any,
    strategy: Any,
) -> dict[str, Any]:
    video_id = video_path.stem
    video_out_dir = out_dir / video_id
    metadata_path = video_out_dir / "metadata.json"

    if args.resume and metadata_path.exists() and not args.overwrite:
        return {
            "video_id": video_id,
            "video_path": str(video_path),
            "out_dir": str(video_out_dir),
            "status": "skipped_existing",
        }

    if args.overwrite and video_out_dir.exists():
        shutil.rmtree(video_out_dir)
    video_out_dir.mkdir(parents=True, exist_ok=True)

    duration_ms, fps, frame_count = video_duration_ms(video_path)
    duration_sec = duration_ms / 1000.0
    n_frames = strategy.context_frames_for(duration_sec)
    extractor = frame_extractor_cls()

    frames = extractor.extract_frames_with_context(
        video_path,
        int(duration_ms),
        n_frames=n_frames,
        context_mode=args.sampling,
        key_moment_threshold_sec=args.key_moment_threshold_sec,
        context_seconds=None,
    )

    saved_frames = []
    for idx, (timestamp_ms, frame) in enumerate(frames, start=1):
        frame_name = f"frame_{idx:03d}_{int(timestamp_ms):06d}ms.{args.frame_format}"
        frame_path = video_out_dir / frame_name
        if args.frame_format == "jpg":
            frame.save(frame_path, quality=args.jpeg_quality)
        else:
            frame.save(frame_path)
        saved_frames.append(
            {
                "index": idx,
                "timestamp_ms": int(timestamp_ms),
                "file": frame_name,
            }
        )

    use_key_moments = (
        args.sampling == "key_moments"
        or (
            args.sampling == "auto"
            and duration_ms > args.key_moment_threshold_sec * 1000
        )
    )
    metadata = {
        "video_id": video_id,
        "video_name": video_path.name,
        "video_path": str(video_path),
        "out_dir": str(video_out_dir),
        "duration_ms": round(duration_ms, 2),
        "fps": round(fps, 3),
        "source_frame_count": frame_count,
        "requested_frames": n_frames,
        "saved_frames": len(saved_frames),
        "sampling": args.sampling,
        "effective_sampling": "key_moments" if use_key_moments else "uniform",
        "key_moment_threshold_sec": args.key_moment_threshold_sec,
        "frame_format": args.frame_format,
        "frames": saved_frames,
        "status": "ok" if saved_frames else "no_frames_saved",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(args.video_dir, args.videos)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise RuntimeError(f"No videos found in {args.video_dir}")

    frame_extractor_cls, base_strategy = load_ck_components()
    strategy = replace(
        base_strategy,
        context_mode=args.sampling,
        key_moment_threshold_sec=args.key_moment_threshold_sec,
    )

    print(f"Video directory: {args.video_dir}", flush=True)
    print(f"Output directory: {args.out_dir}", flush=True)
    print(f"Videos to process: {len(videos)}", flush=True)
    print(
        "Frame rule: ck_whole_video_script STRATEGY_CK "
        f"({strategy.frames_per_sec} fps, "
        f"{strategy.min_context_frames}-{strategy.max_context_frames} frames), "
        f"sampling={args.sampling}",
        flush=True,
    )

    results = []
    failures = 0
    for idx, video_path in enumerate(videos, start=1):
        try:
            metadata = extract_one_video(
                video_path, args.out_dir, args, frame_extractor_cls, strategy
            )
            results.append(metadata)
            print(
                f"[{idx}/{len(videos)}] {video_path.name}: "
                f"{metadata.get('status')} ({metadata.get('saved_frames', 0)} frames)",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            error = {
                "video_id": video_path.stem,
                "video_name": video_path.name,
                "video_path": str(video_path),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(error)
            print(f"[{idx}/{len(videos)}] {video_path.name}: failed: {exc}", flush=True)
            if args.fail_fast:
                break

    summary = {
        "video_dir": str(args.video_dir),
        "out_dir": str(args.out_dir),
        "n_videos_requested": len(videos),
        "n_ok": sum(1 for row in results if row.get("status") == "ok"),
        "n_skipped": sum(1 for row in results if row.get("status") == "skipped_existing"),
        "n_failed": failures,
        "results": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        print(f"Finished with {failures} failures. See {args.out_dir / 'summary.json'}")
        return 1
    print(f"Finished. Summary: {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
