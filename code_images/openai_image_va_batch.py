"""Rate image datasets with OpenAI and probe per-request image-count limits.

The default request size is 50 images. This is intentionally conservative:
the OpenAI API prompt/input window is about 200 MB, and the largest raw images
in these datasets are from NAPS at about 2.5 MB each.

Images are sent from the original files without resizing or recompression.

Outputs are written under code_images/image_va_batch_results/<dataset> by default.
Set OPENAI_API_KEY, and optionally OPENAI_BASE_URL for Navigator/proxy use.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


DATASET_DIRS = {
    "iaps": Path(r"N:\Experimental_Data\yujunchen\projects\data\IAPS1182\IAPS1182"),
    "naps": Path(r"N:\Experimental_Data\yujunchen\projects\data\NAPS_H\NAPS_H"),
    "oasis": Path(r"N:\Experimental_Data\yujunchen\projects\data\OASIS\images"),
}
DATASET_ORDER = ("iaps", "naps", "oasis")
DEFAULT_DATASET = "all"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "image_va_batch_results"

# The OpenAI API prompt/input window is about 200 MB. NAPS contains the largest
# raw images here, about 2.5 MB each, so 50 images/request is a conservative
# default. The script sends original file bytes without resizing or recompressing.
DEFAULT_BATCH_SIZES = "50"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


SYSTEM_PROMPT = (
    "You are a scientific research assistant performing emotion annotation "
    "for an academic study. Your task is to classify the emotional valence "
    "and arousal conveyed in images. Provide structured numerical ratings "
    "only. Do not describe or narrate the image content."
)

USER_INSTRUCTIONS = (
    "You will be shown a SET of images in this single request.\n\n"
    "Rate the emotional content conveyed by each image on two dimensions:\n"
    "- Valence (1.00-9.00): 1=very negative, 5=neutral, 9=very positive\n"
    "- Arousal (1.00-9.00): 1=very calm/low energy, "
    "9=very excited/high energy\n\n"
    "Be precise. Use the FULL continuous scale with two decimal places.\n"
    "Avoid snapping to anchors (.00, .25, .50, .75) or whole numbers unless "
    "truly necessary. Make fine-grained distinctions across the images.\n\n"
    "Rate perceived affect for a typical human observer, not your personal "
    "preference. Because multiple images are shown together, calibrate ratings "
    "relative to the whole set.\n\n"
    "Return ONLY valid JSON with a 'ratings' key containing one object per "
    "image in the same order as the provided image IDs, exactly like:\n"
    '{"ratings":[{"image_id":"2055.1","valence":3.47,"arousal":6.83}]}\n'
    "Do not include any other text or explanation."
)


@dataclass
class ImageItem:
    image_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rate IAPS/NAPS/OASIS images and test OpenAI images/request size."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *sorted(DATASET_DIRS)],
        default=DEFAULT_DATASET,
        help="Named dataset directory to use, or all. Defaults to all.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Override the image directory selected by --dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override the output directory. Defaults to code_images/image_va_batch_results/<dataset>.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.4"),
        help="OpenAI/Navigator model name. Defaults to OPENAI_MODEL or gpt-5.4.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="auto")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resize-max-side",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch-sizes",
        default=DEFAULT_BATCH_SIZES,
        help=(
            "Images per request. If comma-separated values are provided, only "
            "the first value is used for normal full-dataset runs and test mode."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["responses", "batch"],
        default="responses",
        help=(
            "responses sends a live /v1/responses request; batch submits one "
            "JSONL line to the Batch API and polls it."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=20000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build request previews and estimate payload sizes without API calls.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help=(
            "Run exactly one request for each dataset (iaps, naps, oasis). "
            "Uses the first value from --batch-sizes; default is 50."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed size instead of continuing the sweep.",
    )
    args = parser.parse_args()
    args.out_dir_was_provided = args.out_dir is not None
    if args.image_dir is None and args.dataset != "all":
        args.image_dir = DATASET_DIRS[args.dataset]
    if args.out_dir is None and args.dataset != "all":
        args.out_dir = DEFAULT_OUT_ROOT / args.dataset
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_ROOT
    return args


def repair_invalid_cert_env() -> None:
    """Avoid httpx/OpenAI startup crashes from stale certificate env vars.

    Conda/Git Bash environments sometimes leave SSL_CERT_FILE pointing at a
    certificate bundle that was removed when an environment changed. httpx reads
    that variable during OpenAI() construction and raises FileNotFoundError
    before any API call is made. Prefer certifi's current bundle when possible.
    """
    cert_vars = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
    replacement = None

    for var_name in cert_vars:
        value = os.environ.get(var_name)
        if not value:
            continue
        if Path(value).exists():
            continue

        if replacement is None:
            try:
                import certifi

                candidate = certifi.where()
                if candidate and Path(candidate).exists():
                    replacement = candidate
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
                f"[SSL] {var_name} pointed to a missing file; "
                "removed it for this process.",
                flush=True,
            )


def id_to_sort_key(value: str) -> tuple[float, int, str]:
    try:
        return (float(value), 0, value)
    except ValueError:
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return (float(match.group(0)), 0, value)
        return (float("inf"), 1, value)


def extract_image_id(path: Path) -> str:
    # Preserve the exact image filename in prompts and CSV outputs so ratings
    # can be joined back to IAPS, NAPS, and OASIS without losing names.
    return path.name


def list_images(image_dir: Path) -> list[ImageItem]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    items = [
        ImageItem(extract_image_id(path), path)
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    items.sort(key=lambda item: id_to_sort_key(item.image_id))
    return items


def image_to_data_url(path: Path) -> str:
    """Encode original image bytes without resizing or recompression."""
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None:
        media_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
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
                        "image_id": {"type": "string"},
                        "valence": {"type": "number", "minimum": 1, "maximum": 9},
                        "arousal": {"type": "number", "minimum": 1, "maximum": 9},
                    },
                    "required": ["image_id", "valence", "arousal"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["ratings"],
        "additionalProperties": False,
    }


def build_request_body(
    items: list[ImageItem],
    args: argparse.Namespace,
    include_images: bool = True,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": USER_INSTRUCTIONS}]

    for item in items:
        content.append({"type": "input_text", "text": f"image_id: {item.image_id}"})
        if include_images:
            data_url = image_to_data_url(item.path)
        else:
            media_type, _ = mimetypes.guess_type(item.path.name)
            data_url = f"data:{media_type or 'image/jpeg'};base64,<omitted in preview>"
        content.append(
            {"type": "input_image", "image_url": data_url, "detail": args.detail}
        )

    body: dict[str, Any] = {
        "model": args.model,
        "temperature": args.temperature,
        "instructions": SYSTEM_PROMPT,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "image_va_ratings",
                "strict": True,
                "schema": rating_schema(len(items)),
            }
        },
        "input": [{"role": "user", "content": content}],
    }
    if args.max_output_tokens > 0:
        body["max_output_tokens"] = args.max_output_tokens
    return body


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


def write_ratings_csv(path: Path, ratings: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "valence", "arousal"])
        writer.writeheader()
        for row in ratings:
            writer.writerow(
                {
                    "image_name": row.get("image_id"),
                    "valence": row.get("valence"),
                    "arousal": row.get("arousal"),
                }
            )


def validate_ratings(
    ratings: list[dict[str, Any]],
    expected_items: list[ImageItem],
) -> tuple[bool, str]:
    if len(ratings) != len(expected_items):
        return False, f"Expected {len(expected_items)} ratings, got {len(ratings)}."

    expected_ids = [item.image_id for item in expected_items]
    got_ids = [str(row.get("image_id")) for row in ratings]
    if got_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(got_ids), key=id_to_sort_key)
        extra = sorted(set(got_ids) - set(expected_ids), key=id_to_sort_key)
        return False, f"Image IDs/order mismatch. Missing={missing[:10]} Extra={extra[:10]}"

    for row in ratings:
        try:
            valence = float(row["valence"])
            arousal = float(row["arousal"])
        except (KeyError, TypeError, ValueError):
            return False, f"Non-numeric rating row: {row}"
        if not (1 <= valence <= 9 and 1 <= arousal <= 9):
            return False, f"Rating outside 1-9 range: {row}"

    return True, "OK"


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
    size: int,
    ok: bool,
    status: str,
    elapsed_seconds: int,
    request_bytes: int,
    extra: dict[str, Any] | None = None,
) -> None:
    attempt = {
        "size": size,
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


def run_dataset(args: argparse.Namespace, client: Any) -> int:
    sizes = [int(part.strip()) for part in args.batch_sizes.split(",") if part.strip()]
    chunk_size = sizes[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items_all = list_images(args.image_dir)
    if not items_all:
        raise RuntimeError(f"No image files found in {args.image_dir}")

    if len(items_all) < chunk_size:
        raise RuntimeError(
            f"Need {chunk_size} images for one request, found {len(items_all)}."
        )

    one_request_only = bool(getattr(args, "one_request_only", False))
    process_items = items_all[:chunk_size] if one_request_only else items_all

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "image_dir": str(args.image_dir),
        "out_dir": str(args.out_dir),
        "model": args.model,
        "mode": args.mode,
        "detail": args.detail,
        "image_encoding": "original_file_bytes",
        "resized": False,
        "recompressed": False,
        "chunk_size": chunk_size,
        "available_images": len(items_all),
        "total_images": len(process_items),
        "total_chunks": (len(process_items) + chunk_size - 1) // chunk_size,
        "one_request_only": one_request_only,
        "dry_run": args.dry_run,
        "attempts": [],
    }

    preview_body = build_request_body(items_all[: min(3, len(items_all))], args, include_images=False)
    (args.out_dir / "request_preview_first_3.json").write_text(
        json.dumps(preview_body, indent=2), encoding="utf-8"
    )

    base_url = os.environ.get("OPENAI_BASE_URL") or "OpenAI SDK default"
    print(f"\n=== Dataset: {args.dataset} ===", flush=True)
    print(f"Image directory: {args.image_dir}", flush=True)
    print(f"Output directory: {args.out_dir}", flush=True)
    print(f"Using model: {args.model}", flush=True)
    print(f"Using OpenAI base URL: {base_url}", flush=True)
    print(f"Images per request: {chunk_size}", flush=True)
    print(f"Total images to process: {len(process_items)}", flush=True)

    rows_all: list[dict[str, Any]] = []
    ok_chunks = 0
    failed_chunks = 0

    for chunk_idx, start in enumerate(range(0, len(process_items), chunk_size), start=1):
        selected = process_items[start:start + chunk_size]
        end = start + len(selected)
        output_label = f"{args.dataset}_chunk_{chunk_idx:04d}_{start + 1:06d}_{end:06d}"
        chunk_extra = {
            "chunk_index": chunk_idx,
            "start_index": start + 1,
            "end_index": end,
            "n_images": len(selected),
            "first_image": selected[0].image_id,
            "last_image": selected[-1].image_id,
            "output_label": output_label,
        }

        print(
            f"\n[Chunk {chunk_idx}/{summary['total_chunks']}] "
            f"{len(selected)} images ({start + 1}-{end})",
            flush=True,
        )
        try:
            body = build_request_body(selected, args, include_images=True)
            request_bytes = len(json.dumps(body).encode("utf-8"))
            print(f"    request JSON size: {request_bytes / 1024 / 1024:.2f} MB")

            if args.dry_run:
                summarize_attempt(
                    args.out_dir,
                    summary,
                    len(selected),
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
                    len(selected),
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
                    f"{exc.msg} at char {exc.pos}; response_text_chars={len(text)}. "
                    "The request was accepted, but the model did not return a "
                    "complete valid JSON ratings object."
                )
                summarize_attempt(
                    args.out_dir,
                    summary,
                    len(selected),
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
                    len(selected),
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
                write_ratings_csv(args.out_dir / "ratings_all.csv", rows_all)
                summarize_attempt(
                    args.out_dir,
                    summary,
                    len(selected),
                    True,
                    message,
                    elapsed,
                    request_bytes,
                    chunk_extra,
                )
                ok_chunks += 1
                summary["completed_images"] = len(rows_all)
                summary["ok_chunks"] = ok_chunks
                summary["failed_chunks"] = failed_chunks
                (args.out_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                print(f"    success in {elapsed}s; ratings saved.", flush=True)
                print(f"    cumulative ratings: {len(rows_all)}/{len(process_items)}", flush=True)
                continue

            summarize_attempt(
                args.out_dir,
                summary,
                len(selected),
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
                "".join(traceback.format_exception(exc)), encoding="utf-8"
            )
            summarize_attempt(
                args.out_dir,
                summary,
                len(selected),
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

    if args.dry_run:
        print(f"\nDry run complete. See {args.out_dir / 'summary.json'}")
        return 0

    summary["completed_images"] = len(rows_all)
    summary["ok_chunks"] = ok_chunks
    summary["failed_chunks"] = failed_chunks
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failed_chunks:
        print(
            f"\nFinished with failures: ok_chunks={ok_chunks}, "
            f"failed_chunks={failed_chunks}, completed_images={len(rows_all)}/{len(process_items)}",
            flush=True,
        )
        return 1

    print(
        f"\nFinished dataset {args.dataset}: {len(rows_all)}/{len(process_items)} ratings saved "
        f"to {args.out_dir / 'ratings_all.csv'}",
        flush=True,
    )
    return 0


def args_for_dataset(
    args: argparse.Namespace,
    dataset: str,
    batch_size: int,
    one_request_only: bool,
) -> argparse.Namespace:
    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.dataset = dataset
    dataset_args.image_dir = DATASET_DIRS[dataset]
    dataset_args.batch_sizes = str(batch_size)
    dataset_args.one_request_only = one_request_only
    if args.out_dir_was_provided:
        dataset_args.out_dir = Path(args.out_dir) / dataset
    else:
        dataset_args.out_dir = DEFAULT_OUT_ROOT / dataset
    return dataset_args


def main() -> int:
    args = parse_args()
    requested_sizes = [
        int(part.strip()) for part in args.batch_sizes.split(",") if part.strip()
    ]
    if not requested_sizes:
        raise RuntimeError("--batch-sizes must contain at least one integer.")

    client = create_openai_client(args)

    if args.test_mode or args.dataset == "all":
        batch_size = requested_sizes[0]
        one_request_only = args.test_mode
        mode_label = "Test mode" if args.test_mode else "All-datasets mode"
        print(
            f"[{mode_label}] Running "
            f"{'one request per dataset' if one_request_only else 'all images in each dataset'} "
            f"with {batch_size} images/request.",
            flush=True,
        )
        statuses = []
        for dataset in DATASET_ORDER:
            dataset_args = args_for_dataset(args, dataset, batch_size, one_request_only)
            status = run_dataset(dataset_args, client)
            statuses.append(status)
            if status != 0 and args.fail_fast:
                break
        return 0 if all(status == 0 for status in statuses) else 1

    return run_dataset(args, client)


if __name__ == "__main__":
    sys.exit(main())
