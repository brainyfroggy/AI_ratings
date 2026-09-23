"""Rate video datasets with OpenAI using whole-video frame summaries.

Videos are represented as ordered frames spanning each clip. This follows the
CK whole-video pattern: one overall valence/arousal judgment per video, frames
only, no audio.

Outputs are written under code_videos/video_va_batch_results/<dataset> by
default. Set OPENAI_API_KEY, and optionally OPENAI_BASE_URL, in the environment.
The OpenAI client is constructed as OpenAI() so the SDK reads those variables.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, TYPE_CHECKING

import cv2
from PIL import Image

if TYPE_CHECKING:
    from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CK_VIDEO_DIR = Path(
    os.environ.get(
        "CK_VIDEOS_DIR",
        r"N:\Experimental_Data\yujunchen\projects\data\original_ckvideo_data\ckvideo",
    )
)
DATASET_DIRS = {"ck": DEFAULT_CK_VIDEO_DIR}
DEFAULT_DATASET = "ck"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "video_va_batch_results"

VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v",
}
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


SYSTEM_PROMPT = (
    "You are a scientific research assistant performing emotion annotation "
    "for an academic study. Your task is to classify the emotional valence "
    "and arousal conveyed in short video clips. Provide structured numerical "
    "ratings only. Do not describe or narrate the video content."
)

USER_INSTRUCTIONS = (
    "You will be shown a SET of videos in this single request. Each video is "
    "represented by ordered frames sampled from the whole clip, oldest to most "
    "recent.\n\n"
    "Rate the OVERALL emotional content conveyed by each video on two "
    "dimensions:\n"
    "- Valence (1.00-9.00): 1=very negative, 5=neutral, 9=very positive\n"
    "- Arousal (1.00-9.00): 1=very calm/low energy, "
    "9=very excited/high energy\n\n"
    "Use the frames as temporal context to understand how the emotional tone "
    "builds, shifts, or escalates over the clip. Rate perceived affect for a "
    "typical human observer, not your personal preference.\n\n"
    "Be precise. Use the FULL continuous scale with two decimal places. Avoid "
    "snapping to anchors (.00, .25, .50, .75) or whole numbers unless truly "
    "necessary. Because multiple videos are shown together, calibrate ratings "
    "relative to the whole set.\n\n"
    "Return ONLY valid JSON with a 'ratings' key containing one object per "
    "video in the same order as the provided video IDs, exactly like:\n"
    '{"ratings":[{"video_id":"0001.mp4","valence":3.47,"arousal":6.83}]}\n'
    "Do not include any other text or explanation."
)


@dataclass
class VideoItem:
    video_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rate whole videos for valence/arousal with OpenAI."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DIRS),
        default=DEFAULT_DATASET,
        help="Named video dataset. Defaults to ck.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Override the video directory selected by --dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory. Defaults to code_videos/video_va_batch_results/<dataset>.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.4"),
        help="OpenAI/Navigator model name. Defaults to OPENAI_MODEL or gpt-5.4.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="low")
    parser.add_argument(
        "--videos-per-request",
        type=int,
        default=5,
        help="Number of videos per Responses API request. Defaults to 5.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="Minimum frames sampled per video. Defaults to 3.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Maximum frames sampled per video. Defaults to 20.",
    )
    parser.add_argument(
        "--frames-per-sec",
        type=float,
        default=1.0,
        help="Frame budget before min/max clamp. Defaults to 1 frame/sec.",
    )
    parser.add_argument(
        "--sampling",
        choices=["auto", "uniform", "key_moments"],
        default="auto",
        help="Frame selection policy. auto matches CK: uniform up to 10s, key moments after.",
    )
    parser.add_argument(
        "--key-moment-threshold-sec",
        type=float,
        default=10.0,
        help="auto switches to key-moment sampling above this duration.",
    )
    parser.add_argument(
        "--frame-format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Encoding for extracted frames sent to the API. Defaults to jpeg.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--mode",
        choices=["responses", "batch"],
        default="responses",
        help="responses sends live calls; batch submits one JSONL line and polls it.",
    )
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Optional subset by exact filename or stem, e.g. 0001.mp4 or 0001.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run exactly one request, using --videos-per-request videos.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    args.out_dir_was_provided = args.out_dir is not None
    if args.video_dir is None:
        args.video_dir = DATASET_DIRS[args.dataset]
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_ROOT / args.dataset
    if args.videos_per_request < 1:
        raise ValueError("--videos-per-request must be >= 1.")
    if args.min_frames < 1 or args.max_frames < args.min_frames:
        raise ValueError("--min-frames/--max-frames are inconsistent.")
    return args


def repair_invalid_cert_env() -> None:
    """Avoid httpx/OpenAI startup crashes from stale certificate env vars."""
    cert_vars = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
    replacement = None

    for var_name in cert_vars:
        value = os.environ.get(var_name)
        if not value or Path(value).exists():
            continue

        if replacement is None:
            try:
                import certifi

                candidate = certifi.where()
                replacement = candidate if candidate and Path(candidate).exists() else ""
            except Exception:
                replacement = ""

        if replacement:
            os.environ[var_name] = replacement
            print(
                f"[SSL] {var_name} pointed to a missing file; "
                f"using certifi bundle: {replacement}",
                flush=True,
            )
        else:
            os.environ.pop(var_name, None)
            print(
                f"[SSL] {var_name} pointed to a missing file; removed it.",
                flush=True,
            )


def id_to_sort_key(value: str) -> tuple[float, int, str]:
    try:
        return (float(Path(value).stem), 0, value)
    except ValueError:
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return (float(match.group(0)), 0, value)
        return (float("inf"), 1, value)


def list_videos(video_dir: Path) -> list[VideoItem]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    items = [
        VideoItem(path.name, path)
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]
    items.sort(key=lambda item: id_to_sort_key(item.video_id))
    return items


def filter_videos(items: list[VideoItem], requested: list[str] | None) -> list[VideoItem]:
    if not requested:
        return items
    wanted = {value.lower() for value in requested}
    return [
        item
        for item in items
        if item.video_id.lower() in wanted or item.path.stem.lower() in wanted
    ]


def video_metadata(path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            return 5000.0, fps or 30.0, frame_count
        return (frame_count / fps) * 1000.0, fps, frame_count
    finally:
        cap.release()


def frame_count_for_duration(duration_ms: float, args: argparse.Namespace) -> int:
    duration_sec = max(0.001, duration_ms / 1000.0)
    n_frames = int(round(duration_sec * args.frames_per_sec))
    return max(args.min_frames, min(args.max_frames, n_frames))


def extract_frames_with_project_sampler(
    path: Path,
    duration_ms: float,
    n_frames: int,
    args: argparse.Namespace,
) -> list[Image.Image]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.frame_extractor import FrameExtractor

    extractor = FrameExtractor()
    frame_data = extractor.extract_frames_with_context(
        path,
        int(duration_ms),
        n_frames=n_frames,
        context_mode=args.sampling,
        key_moment_threshold_sec=args.key_moment_threshold_sec,
        context_seconds=None,
    )
    return [frame for _, frame in frame_data]


def extract_frames_uniform(
    path: Path,
    duration_ms: float,
    fps: float,
    frame_count: int,
    n_frames: int,
) -> list[Image.Image]:
    if frame_count > 1 and fps > 0:
        last_readable_ms = ((frame_count - 1) / fps) * 1000.0
        end_ms = min(duration_ms, last_readable_ms)
    else:
        end_ms = max(0.0, duration_ms)

    if n_frames == 1:
        timestamps = [end_ms]
    else:
        timestamps = [
            round((end_ms * idx) / (n_frames - 1))
            for idx in range(n_frames)
        ]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        frames: list[Image.Image] = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        if not frames:
            raise ValueError(f"No frames could be read from {path}")
        return frames
    finally:
        cap.release()


def extract_video_frames(
    item: VideoItem,
    args: argparse.Namespace,
) -> tuple[list[Image.Image], dict[str, Any]]:
    duration_ms, fps, frame_count = video_metadata(item.path)
    n_frames = frame_count_for_duration(duration_ms, args)
    sampler = args.sampling

    try:
        frames = extract_frames_with_project_sampler(item.path, duration_ms, n_frames, args)
    except Exception as exc:
        if sampler == "key_moments":
            raise
        frames = extract_frames_uniform(item.path, duration_ms, fps, frame_count, n_frames)
        sampler = f"uniform_fallback_after_{type(exc).__name__}"

    meta = {
        "duration_ms": round(duration_ms, 2),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "requested_frames": n_frames,
        "sent_frames": len(frames),
        "sampling": sampler,
    }
    return frames, meta


def frame_to_data_url(frame: Image.Image, args: argparse.Namespace) -> str:
    buffer = BytesIO()
    if args.frame_format == "png":
        frame.save(buffer, format="PNG")
        media_type = "image/png"
    else:
        frame.save(buffer, format="JPEG", quality=args.jpeg_quality)
        media_type = "image/jpeg"
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"


def rating_schema(expected_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ratings": {
                "type": "array",
                "minItems": expected_count,
                "maxItems": expected_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string"},
                        "valence": {"type": "number", "minimum": 1, "maximum": 9},
                        "arousal": {"type": "number", "minimum": 1, "maximum": 9},
                    },
                    "required": ["video_id", "valence", "arousal"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["ratings"],
        "additionalProperties": False,
    }


def build_request_body(
    items: list[VideoItem],
    args: argparse.Namespace,
    include_images: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": USER_INSTRUCTIONS}]
    video_meta: list[dict[str, Any]] = []

    for item in items:
        frames, meta = extract_video_frames(item, args)
        meta.update({"video_id": item.video_id, "path": str(item.path)})
        video_meta.append(meta)
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"video_id: {item.video_id}\n"
                    f"frames: {len(frames)} ordered frames spanning the full clip"
                ),
            }
        )
        for idx, frame in enumerate(frames, start=1):
            content.append(
                {
                    "type": "input_text",
                    "text": f"{item.video_id} frame {idx} of {len(frames)}",
                }
            )
            if include_images:
                image_url = frame_to_data_url(frame, args)
            else:
                image_url = f"data:image/{args.frame_format};base64,<omitted>"
            content.append(
                {"type": "input_image", "image_url": image_url, "detail": args.detail}
            )

    body: dict[str, Any] = {
        "model": args.model,
        "temperature": args.temperature,
        "instructions": SYSTEM_PROMPT,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "video_va_ratings",
                "strict": True,
                "schema": rating_schema(len(items)),
            }
        },
        "input": [{"role": "user", "content": content}],
    }
    if args.max_output_tokens > 0:
        body["max_output_tokens"] = args.max_output_tokens
    return body, video_meta


def response_to_output_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str):
        return text

    if hasattr(resp, "model_dump"):
        body = resp.model_dump()
    elif isinstance(resp, dict):
        body = resp
    else:
        return ""

    if isinstance(body.get("output_text"), str):
        return body["output_text"]

    chunks: list[str] = []
    for item in body.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and "text" in content:
                chunks.append(content["text"])
    return "\n".join(chunks)


def parse_ratings(text: str) -> list[dict[str, Any]]:
    parsed = json.loads(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("ratings"), list):
        return parsed["ratings"]
    raise ValueError("Response JSON did not contain a ratings array.")


def validate_ratings(
    ratings: list[dict[str, Any]],
    expected_items: list[VideoItem],
) -> tuple[bool, str]:
    if len(ratings) != len(expected_items):
        return False, f"Expected {len(expected_items)} ratings, got {len(ratings)}."

    expected_ids = [item.video_id for item in expected_items]
    got_ids = [str(row.get("video_id")) for row in ratings]
    if got_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(got_ids), key=id_to_sort_key)
        extra = sorted(set(got_ids) - set(expected_ids), key=id_to_sort_key)
        return False, f"Video IDs/order mismatch. Missing={missing[:10]} Extra={extra[:10]}"

    for row in ratings:
        try:
            valence = float(row["valence"])
            arousal = float(row["arousal"])
        except (KeyError, TypeError, ValueError):
            return False, f"Non-numeric rating row: {row}"
        if not (1 <= valence <= 9 and 1 <= arousal <= 9):
            return False, f"Rating outside 1-9 range: {row}"

    return True, "OK"


def write_ratings_csv(path: Path, ratings: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stimulus_id", "video_name", "valence", "arousal"],
        )
        writer.writeheader()
        for row in ratings:
            video_name = str(row.get("video_id"))
            writer.writerow(
                {
                    "stimulus_id": Path(video_name).stem,
                    "video_name": video_name,
                    "valence": row.get("valence"),
                    "arousal": row.get("arousal"),
                }
            )


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row.get("video_name", "")
            for row in reader
            if row.get("video_name") and row.get("valence") and row.get("arousal")
        }


def call_responses(
    client: "OpenAI",
    body: dict[str, Any],
    out_dir: Path,
    output_label: str,
) -> tuple[bool, str, int]:
    started = time.time()
    resp = client.responses.create(**body)
    elapsed = round(time.time() - started)

    raw_path = out_dir / f"responses_raw_{output_label}.json"
    if hasattr(resp, "model_dump_json"):
        raw_path.write_text(resp.model_dump_json(indent=2), encoding="utf-8")
    else:
        raw_path.write_text(json.dumps(resp, indent=2), encoding="utf-8")

    text = response_to_output_text(resp)
    (out_dir / f"responses_text_{output_label}.json").write_text(text, encoding="utf-8")
    return True, text, elapsed


def wait_for_batch(client: "OpenAI", batch_id: str, poll_seconds: int) -> Any:
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"    batch {batch_id} status={batch.status}", flush=True)
        if batch.status in TERMINAL_BATCH_STATUSES:
            return batch
        time.sleep(poll_seconds)


def call_batch(
    client: "OpenAI",
    body: dict[str, Any],
    out_dir: Path,
    output_label: str,
    poll_seconds: int,
) -> tuple[bool, str, int]:
    started = time.time()
    request_path = out_dir / f"batch_request_{output_label}.jsonl"
    line = {
        "custom_id": output_label,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }
    request_path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    with request_path.open("rb") as handle:
        batch_file = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    (out_dir / f"batch_id_{output_label}.txt").write_text(batch.id, encoding="utf-8")

    final = wait_for_batch(client, batch.id, poll_seconds)
    elapsed = round(time.time() - started)
    if final.status != "completed":
        error_text = json.dumps(getattr(final, "errors", None), default=str)
        return False, f"Batch ended with status={final.status}. errors={error_text}", elapsed

    if getattr(final, "error_file_id", None):
        error_bytes = client.files.content(final.error_file_id).read()
        error_path = out_dir / f"batch_errors_{output_label}.jsonl"
        error_path.write_bytes(error_bytes)
        return False, error_path.read_text(encoding="utf-8", errors="replace"), elapsed

    if not getattr(final, "output_file_id", None):
        return False, "Batch completed without output_file_id.", elapsed

    output_bytes = client.files.content(final.output_file_id).read()
    output_path = out_dir / f"batch_output_{output_label}.jsonl"
    output_path.write_bytes(output_bytes)

    first_line = output_path.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    body = ((record.get("response") or {}).get("body") or {})
    return True, response_to_output_text(body), elapsed


def summarize_attempt(
    out_dir: Path,
    summary: dict[str, Any],
    ok: bool,
    status: str,
    elapsed_seconds: int,
    request_bytes: int,
    extra: dict[str, Any] | None = None,
) -> None:
    attempt = {
        "ok": ok,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "request_json_bytes": request_bytes,
    }
    if extra:
        attempt.update(extra)
    summary["attempts"].append(attempt)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def create_openai_client(args: argparse.Namespace) -> Any:
    if args.dry_run:
        return None

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The OpenAI Python SDK is required for live API calls. "
            "Install it with: pip install openai"
        ) from exc

    repair_invalid_cert_env()
    return OpenAI()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items_all = list_videos(args.video_dir)
    items_all = filter_videos(items_all, args.videos)
    if args.limit is not None:
        items_all = items_all[: args.limit]
    if not items_all:
        raise RuntimeError(f"No video files found in {args.video_dir}")

    ratings_all_path = args.out_dir / "ratings_all.csv"
    rows_all: list[dict[str, Any]] = []
    if args.resume and ratings_all_path.exists():
        with ratings_all_path.open("r", newline="", encoding="utf-8") as handle:
            rows_all = [
                {
                    "video_id": row.get("video_name") or row.get("video_id"),
                    "valence": row.get("valence"),
                    "arousal": row.get("arousal"),
                }
                for row in csv.DictReader(handle)
            ]
        completed = read_completed_ids(ratings_all_path)
        items_all = [item for item in items_all if item.video_id not in completed]
        print(f"Resume: {len(completed)} videos already done; {len(items_all)} to go.")

    if args.test_mode:
        items_all = items_all[: args.videos_per_request]

    client = create_openai_client(args)

    preview_items = items_all[: min(2, len(items_all))]
    preview_body, preview_meta = build_request_body(preview_items, args, include_images=False)
    (args.out_dir / "request_preview_first_2.json").write_text(
        json.dumps(preview_body, indent=2), encoding="utf-8"
    )
    (args.out_dir / "frame_preview_first_2.json").write_text(
        json.dumps(preview_meta, indent=2), encoding="utf-8"
    )

    total_chunks = (len(items_all) + args.videos_per_request - 1) // args.videos_per_request
    base_url = os.environ.get("OPENAI_BASE_URL") or "OpenAI SDK default"
    print(f"\n=== Dataset: {args.dataset} ===", flush=True)
    print(f"Video directory: {args.video_dir}", flush=True)
    print(f"Output directory: {args.out_dir}", flush=True)
    print(f"Using model: {args.model}", flush=True)
    print(f"Using OpenAI base URL: {base_url}", flush=True)
    print(f"Videos per request: {args.videos_per_request}", flush=True)
    print(f"Total videos to process: {len(items_all)}", flush=True)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "video_dir": str(args.video_dir),
        "out_dir": str(args.out_dir),
        "model": args.model,
        "mode": args.mode,
        "detail": args.detail,
        "videos_per_request": args.videos_per_request,
        "frame_budget": {
            "frames_per_sec": args.frames_per_sec,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "sampling": args.sampling,
            "key_moment_threshold_sec": args.key_moment_threshold_sec,
        },
        "frame_encoding": args.frame_format,
        "available_videos": len(items_all),
        "total_chunks": total_chunks,
        "test_mode": args.test_mode,
        "dry_run": args.dry_run,
        "attempts": [],
    }

    ok_chunks = 0
    failed_chunks = 0

    for chunk_idx, start in enumerate(
        range(0, len(items_all), args.videos_per_request),
        start=1,
    ):
        selected = items_all[start:start + args.videos_per_request]
        end = start + len(selected)
        output_label = f"{args.dataset}_chunk_{chunk_idx:04d}_{start + 1:06d}_{end:06d}"
        chunk_extra = {
            "chunk_index": chunk_idx,
            "start_index": start + 1,
            "end_index": end,
            "n_videos": len(selected),
            "first_video": selected[0].video_id,
            "last_video": selected[-1].video_id,
            "output_label": output_label,
        }

        print(
            f"\n[Chunk {chunk_idx}/{total_chunks}] "
            f"{len(selected)} videos ({start + 1}-{end})",
            flush=True,
        )
        try:
            body, video_meta = build_request_body(selected, args, include_images=True)
            chunk_extra["video_meta"] = video_meta
            request_bytes = len(json.dumps(body).encode("utf-8"))
            print(f"    request JSON size: {request_bytes / 1024 / 1024:.2f} MB")

            if args.dry_run:
                summarize_attempt(
                    args.out_dir,
                    summary,
                    True,
                    "dry_run_request_built",
                    0,
                    request_bytes,
                    chunk_extra,
                )
                continue

            assert client is not None
            if args.mode == "responses":
                ok, text, elapsed = call_responses(client, body, args.out_dir, output_label)
            else:
                ok, text, elapsed = call_batch(
                    client, body, args.out_dir, output_label, args.poll_seconds
                )

            if not ok:
                summarize_attempt(
                    args.out_dir,
                    summary,
                    False,
                    text[:1000],
                    elapsed,
                    request_bytes,
                    chunk_extra,
                )
                failed_chunks += 1
                print(f"    failed: {text[:500]}", flush=True)
                if args.fail_fast:
                    break
                continue

            try:
                ratings = parse_ratings(text)
            except json.JSONDecodeError as exc:
                message = (
                    "malformed_or_truncated_json: "
                    f"{exc.msg} at char {exc.pos}; response_text_chars={len(text)}"
                )
                summarize_attempt(
                    args.out_dir,
                    summary,
                    False,
                    message,
                    elapsed,
                    request_bytes,
                    chunk_extra,
                )
                failed_chunks += 1
                print(f"    response JSON failed: {message}", flush=True)
                if args.fail_fast:
                    break
                continue
            except ValueError as exc:
                message = f"invalid_response_json_shape: {exc}; response_text_chars={len(text)}"
                summarize_attempt(
                    args.out_dir,
                    summary,
                    False,
                    message,
                    elapsed,
                    request_bytes,
                    chunk_extra,
                )
                failed_chunks += 1
                print(f"    response JSON failed: {message}", flush=True)
                if args.fail_fast:
                    break
                continue

            valid, message = validate_ratings(ratings, selected)
            if valid:
                rows_all.extend(ratings)
                write_ratings_csv(args.out_dir / f"ratings_{output_label}.csv", ratings)
                write_ratings_csv(ratings_all_path, rows_all)
                ok_chunks += 1
                summary["completed_videos"] = len(rows_all)
                summary["ok_chunks"] = ok_chunks
                summary["failed_chunks"] = failed_chunks
                summarize_attempt(
                    args.out_dir,
                    summary,
                    True,
                    message,
                    elapsed,
                    request_bytes,
                    chunk_extra,
                )
                print(f"    success in {elapsed}s; ratings saved.", flush=True)
                print(f"    cumulative ratings: {len(rows_all)}/{len(items_all)}", flush=True)
                continue

            summarize_attempt(
                args.out_dir,
                summary,
                False,
                message,
                elapsed,
                request_bytes,
                chunk_extra,
            )
            failed_chunks += 1
            print(f"    response validation failed: {message}", flush=True)
            if args.fail_fast:
                break

        except Exception as exc:
            error_path = args.out_dir / f"error_{output_label}.txt"
            error_path.write_text(
                "".join(traceback.format_exception(exc)),
                encoding="utf-8",
            )
            summarize_attempt(
                args.out_dir,
                summary,
                False,
                f"{type(exc).__name__}: {exc}",
                0,
                0,
                chunk_extra,
            )
            failed_chunks += 1
            print(f"    exception: {type(exc).__name__}: {exc}", flush=True)
            if args.fail_fast:
                break

    summary["completed_videos"] = len(rows_all)
    summary["ok_chunks"] = ok_chunks
    summary["failed_chunks"] = failed_chunks
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"\nDry run complete. See {args.out_dir / 'summary.json'}")
        return 0

    if failed_chunks:
        print(
            f"\nFinished with failures: ok_chunks={ok_chunks}, "
            f"failed_chunks={failed_chunks}, completed_videos={len(rows_all)}/{len(items_all)}",
            flush=True,
        )
        return 1

    print(
        f"\nFinished dataset {args.dataset}: {len(rows_all)}/{len(items_all)} ratings saved "
        f"to {ratings_all_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
