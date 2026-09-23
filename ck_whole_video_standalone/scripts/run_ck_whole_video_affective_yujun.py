#!/usr/bin/env python3
r"""Extract CK whole-video frames and rate 14 affective dimensions.

This is the Yujun variant of the validated CK whole-video runner. It keeps the
existing frame-extraction/rating code untouched and sends one request per video.
For each video it first uses the validated CK frame-selection algorithm to
generate frames, saves those frames, then sends the generated frame files to
OpenAI without resizing or re-encoding them during the API request.

Default input:
    N:\Experimental_Data\yujunchen\projects\data\original_ckvideo_data\ckvideo

Default output:
    ck_whole_video_standalone/output/ck/ratings/<llm>_affective14_yujun.csv
"""

from __future__ import annotations

import argparse
import base64
import cv2
import json
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ck_config import CKConfig
from src.ck_dimensions import CK_DIMENSIONS, DIMENSION_DEFINITIONS
from src.config import Config
from src.frame_extractor import FrameExtractor
from src.rating_strategy import STRATEGY_CK


AI_RATINGS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\original_ckvideo_data\ckvideo"
)
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}


class RateLimiter:
    """Thread-safe global requests-per-minute limiter."""

    def __init__(self, rpm: float):
        self.delay = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.lock = threading.Lock()
        self.next_t = 0.0

    def acquire(self) -> None:
        if self.delay <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait = self.next_t - now
            if wait > 0:
                time.sleep(wait)
            self.next_t = max(now, self.next_t) + self.delay


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "CK whole-video OpenAI rating for the 14 Cowen-Keltner affective "
            "dimensions only."
        )
    )
    ap.add_argument("--llm", default="gpt-5.4", help="Output label and default model id.")
    ap.add_argument("--model", default=None, help="OpenAI model id. Defaults to --llm.")
    ap.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Folder containing original CK video files.",
    )
    ap.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Where generated frames are saved. Defaults to output/ck/frames_yujun.",
    )
    ap.add_argument(
        "--ratings-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to CKConfig.ratings_output_dir.",
    )
    ap.add_argument("--videos", nargs="*", default=None, help="Subset of video ids.")
    ap.add_argument("--image-detail", choices=["low", "high", "auto"], default="low")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--rpm", type=float, default=30.0)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Regenerate frame folders even if metadata.json already exists.",
    )
    ap.add_argument(
        "--test-mode",
        action="store_true",
        help="Rate only the first requested video. This still makes one API call.",
    )
    return ap.parse_args()


def list_frame_files(video_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )


def id_sort_key(video_id: str) -> tuple[float, int, str]:
    try:
        return (float(video_id), 0, video_id)
    except ValueError:
        return (float("inf"), 1, video_id)


def list_video_ids(video_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return [Path(v).stem for v in requested]
    videos = [
        p.stem
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    return sorted(videos, key=id_sort_key)


def video_duration_ms(path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            return 5000.0, fps, frame_count
        return ((frame_count - 1) / fps) * 1000.0, fps, frame_count
    finally:
        cap.release()


def get_video_path(video_dir: Path, video_id: str) -> Path:
    requested = Path(video_id)
    if requested.suffix:
        path = video_dir / requested.name
        if path.exists():
            return path
    for ext in sorted(VIDEO_EXTS):
        path = video_dir / f"{Path(video_id).stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Video not found for {video_id} in {video_dir}")


def extract_or_load_frames(
    video_id: str,
    video_path: Path,
    frames_root: Path,
    overwrite: bool,
) -> tuple[list[Path], dict[str, Any]]:
    out_dir = frames_root / video_id
    metadata_path = out_dir / "metadata.json"

    if not overwrite and metadata_path.exists():
        frame_files = list_frame_files(out_dir)
        if frame_files:
            metadata = json.loads(metadata_path.read_text())
            return frame_files, metadata

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    for old in out_dir.glob("frame_*.png"):
        old.unlink()

    duration_ms, fps, frame_count = video_duration_ms(video_path)
    duration_sec = duration_ms / 1000.0
    requested_frames = STRATEGY_CK.context_frames_for(duration_sec)

    extractor = FrameExtractor(Config())
    frame_data = extractor.extract_frames_with_context(
        video_path,
        int(duration_ms),
        n_frames=requested_frames,
        context_mode=STRATEGY_CK.context_mode,
        key_moment_threshold_sec=STRATEGY_CK.key_moment_threshold_sec,
        context_seconds=STRATEGY_CK.context_seconds,
    )

    saved = []
    for idx, (ts, frame) in enumerate(frame_data, start=1):
        path = out_dir / f"frame_{idx:03d}_{ts:06d}ms.jpg"
        frame.save(path, quality=95)
        saved.append(path)

    metadata = {
        "video_id": video_id,
        "video_path": str(video_path),
        "duration_ms": round(duration_ms, 3),
        "duration_sec": round(duration_sec, 6),
        "fps": fps,
        "source_frame_count": frame_count,
        "requested_frames": requested_frames,
        "saved_frames": len(saved),
        "frame_timestamps_ms": [ts for ts, _ in frame_data],
        "sampling": STRATEGY_CK.context_mode,
        "key_moment_threshold_sec": STRATEGY_CK.key_moment_threshold_sec,
        "frames_per_sec": STRATEGY_CK.frames_per_sec,
        "min_context_frames": STRATEGY_CK.min_context_frames,
        "max_context_frames": STRATEGY_CK.max_context_frames,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return list_frame_files(out_dir), metadata


def image_file_to_content(path: Path, detail: str) -> dict[str, Any]:
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{data}",
            "detail": detail,
        },
    }


def build_affective14_prompt(video_id: str, n_frames: int) -> str:
    lines = [
        "You are a scientific research assistant performing emotion annotation "
        "for an academic study.",
        "",
        f"You will see {n_frames} key frames from one short video clip "
        f"({video_id}), ordered from oldest to most recent.",
        "Use the frames as temporal context and rate the OVERALL emotional "
        "content of the whole clip.",
        "",
        "Rate all 14 Cowen-Keltner affective dimensions on a 1.00-9.00 scale.",
        "Use the full continuous scale with two decimal places.",
        "Do not describe the video content.",
        "",
        "Dimensions and anchors:",
    ]
    for dim in CK_DIMENSIONS:
        lines.append(f"- {dim}: {DIMENSION_DEFINITIONS[dim]}")
    example = ", ".join(f'"{dim}": 5.00' for dim in CK_DIMENSIONS[:3])
    lines.extend(
        [
            "",
            "Respond ONLY with valid JSON in this exact shape:",
            "{",
            f'  "dimensions": {{{example}, ...}}',
            "}",
            "",
            "Include every one of the 14 dimension keys exactly as listed.",
        ]
    )
    return "\n".join(lines)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group())


def parse_affective14_response(text: str) -> dict[str, float]:
    data = extract_json_object(text)
    raw_dims = data.get("dimensions", data)
    if not isinstance(raw_dims, dict):
        raise ValueError("Response does not contain a dimensions object")

    lower_to_key = {str(k).strip().lower(): k for k in raw_dims.keys()}
    result = {}
    missing = []
    for dim in CK_DIMENSIONS:
        raw_key = lower_to_key.get(dim.lower())
        if raw_key is None:
            missing.append(dim)
            continue
        value = float(raw_dims[raw_key])
        result[dim] = round(max(1.0, min(9.0, value)), 2)
    if missing:
        raise ValueError(f"Missing dimensions: {missing}")
    return result


def call_openai_with_fallback(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    try:
        response = client.chat.completions.create(max_tokens=max_tokens, **kwargs)
    except Exception as exc:
        if "max_tokens" not in str(exc) or "max_completion_tokens" not in str(exc):
            raise
        response = client.chat.completions.create(
            max_completion_tokens=max_tokens,
            **kwargs,
        )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError(f"Empty response from {model}")
    return content


def rate_one_video(
    video_id: str,
    video_dir: Path,
    frames_dir: Path,
    client: OpenAI,
    model: str,
    llm_label: str,
    image_detail: str,
    temperature: float,
    limiter: RateLimiter,
    overwrite_frames: bool,
) -> dict[str, Any]:
    video_path = get_video_path(video_dir, video_id)
    frame_files, metadata = extract_or_load_frames(
        video_id=video_id,
        video_path=video_path,
        frames_root=frames_dir,
        overwrite=overwrite_frames,
    )
    if not frame_files:
        raise FileNotFoundError(f"No frames generated for: {video_path}")

    content = [image_file_to_content(p, image_detail) for p in frame_files]
    content.append(
        {
            "type": "text",
            "text": build_affective14_prompt(video_id, len(frame_files)),
        }
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a scientific research assistant. Provide structured "
                "numerical ratings only."
            ),
        },
        {"role": "user", "content": content},
    ]

    limiter.acquire()
    raw = call_openai_with_fallback(
        client=client,
        model=model,
        messages=messages,
        max_tokens=800,
        temperature=temperature,
    )
    dims = parse_affective14_response(raw)
    row = {
        "stimulus_id": video_id,
        "llm": llm_label,
        "n_frames_sent": len(frame_files),
        "requested_frames": metadata.get("requested_frames"),
        "saved_frames": metadata.get("saved_frames"),
        "duration_sec": metadata.get("duration_sec"),
        "success": True,
        "error": "",
    }
    row.update(dims)
    return row


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    ck = CKConfig()
    ck.ensure_directories()

    video_dir = args.video_dir
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    frames_dir = args.frames_dir or (ck.output_dir / "frames_yujun")
    frames_dir.mkdir(parents=True, exist_ok=True)

    videos = list_video_ids(video_dir, args.videos)
    if args.test_mode:
        videos = videos[:1]

    out_dir = args.ratings_dir or ck.ratings_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.llm}_affective14_yujun.csv"

    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if args.resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        if "success" in prev.columns:
            done = set(prev[prev["success"] == True]["stimulus_id"].astype(str))
        print(f"Resume: {len(done)} videos already done; {len(videos) - len(done)} to go.")
    todo = [v for v in videos if v not in done]

    client = OpenAI()
    model = args.model or args.llm
    limiter = RateLimiter(args.rpm)

    print(
        "CK whole-video affective14 Yujun run | "
        f"model={model} | videos={len(todo)}/{len(videos)} | "
        f"video_dir={video_dir} | frames_dir={frames_dir} | output={out_csv}"
    )

    def flush() -> None:
        ordered = [
            "stimulus_id",
            "llm",
            "n_frames_sent",
            "requested_frames",
            "saved_frames",
            "duration_sec",
            "success",
            "error",
        ] + CK_DIMENSIONS
        pd.DataFrame(rows).reindex(columns=ordered).to_csv(out_csv, index=False)

    def handle_video(video_id: str) -> dict[str, Any]:
        try:
            return rate_one_video(
                video_id=video_id,
                video_dir=video_dir,
                frames_dir=frames_dir,
                client=client,
                model=model,
                llm_label=args.llm,
                image_detail=args.image_detail,
                temperature=args.temperature,
                limiter=limiter,
                overwrite_frames=args.overwrite_frames,
            )
        except Exception as exc:
            row = {
                "stimulus_id": video_id,
                "llm": args.llm,
                "n_frames_sent": 0,
                "requested_frames": float("nan"),
                "saved_frames": float("nan"),
                "duration_sec": float("nan"),
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            for dim in CK_DIMENSIONS:
                row[dim] = float("nan")
            return row

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(handle_video, v): v for v in todo}
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
                rows.append(future.result())
                if (i + 1) % args.save_every == 0:
                    flush()
    else:
        for i, video_id in enumerate(tqdm(todo)):
            rows.append(handle_video(video_id))
            if (i + 1) % args.save_every == 0:
                flush()

    flush()
    ok = sum(1 for r in rows if r.get("success"))
    print(f"Done. {ok}/{len(rows)} successful -> {out_csv}")


if __name__ == "__main__":
    main()
