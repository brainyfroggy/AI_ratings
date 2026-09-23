#!/usr/bin/env python3
"""Rate NAPS/IAPS/OASIS still images for valence and arousal.

This runner is intentionally separate from the CK whole-video runner.  It sends
one multimodal request per dataset by default, with every image in that dataset
attached in filename order.  Original filenames are preserved in the output CSV
so ratings can be joined back to the source image files.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


DEFAULT_DATASETS = {
    "NAPS": Path(os.environ.get("NAPS_IMAGES_DIR", "/blue/mzding/yujunchen/projects/data/NAPS_H/NAPS_H")),
    "IAPS": Path(os.environ.get("IAPS_IMAGES_DIR", "/blue/mzding/yujunchen/projects/data/IAPS1182/IAPS1182")),
    "OASIS": Path(os.environ.get("OASIS_IMAGES_DIR", "/blue/mzding/yujunchen/projects/data/OASIS/images")),
}


SYSTEM_MESSAGE = (
    "You are a scientific research assistant performing emotion annotation "
    "for an academic study. Your task is to classify the emotional valence "
    "and arousal conveyed in still images. Provide structured numerical "
    "ratings only. Do not describe or narrate the image content."
)


def build_prompt(dataset: str, items: list["ImageItem"]) -> str:
    manifest = "\n".join(
        f"- {i + 1}: id={item.stimulus_id}; file_name={item.file_name}"
        for i, item in enumerate(items)
    )
    return f"""You will rate one still-image dataset named {dataset}.

You will receive {len(items)} images. Each image is preceded by a text label with its stimulus_id and original file_name. The list below gives the expected order:
{manifest}

Rate the OVERALL emotional content of each image on two dimensions:

- Valence (1.00-9.00): 1=very negative, 5=neutral, 9=very positive
- Arousal (1.00-9.00): 1=very calm/low energy, 9=very excited/high energy

Be precise - use the FULL continuous scale with two decimal places.
Avoid snapping to anchors (.00, .25, .50, .75) or multiples of 5 or 10.
Make fine-grained distinctions.

Respond ONLY with valid JSON in this exact shape:
{{
  "dataset": "{dataset}",
  "ratings": [
    {{"stimulus_id": "original-id", "file_name": "original-file-name.jpg", "valence": 5.00, "arousal": 5.00}}
  ]
}}

Include exactly one rating object for every image listed above. Do not include explanations, markdown, or image descriptions."""


@dataclass(frozen=True)
class ImageItem:
    dataset: str
    stimulus_id: str
    file_name: str
    path: Path


def list_images(dataset: str, directory: Path) -> list[ImageItem]:
    files = sorted(
        (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p.relative_to(directory)).lower(),
    )
    return [
        ImageItem(
            dataset=dataset,
            stimulus_id=str(p.relative_to(directory)).replace("\\", "/"),
            file_name=p.name,
            path=p,
        )
        for p in files
    ]


def image_to_data_url(path: Path, max_side: int, jpeg_quality: int) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def make_content(items: list[ImageItem], prompt: str, image_detail: str, max_side: int, jpeg_quality: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for item in items:
        content.append({
            "type": "text",
            "text": f"stimulus_id={item.stimulus_id}; file_name={item.file_name}",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(item.path, max_side=max_side, jpeg_quality=jpeg_quality),
                "detail": image_detail,
            },
        })
    return content


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
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
        return json.loads(match.group(0))


def coerce_rating(value: Any) -> float:
    val = float(value)
    return round(max(1.0, min(9.0, val)), 2)


def normalize_rows(dataset: str, items: list[ImageItem], response_text: str, model: str) -> list[dict[str, Any]]:
    data = parse_json_object(response_text)
    ratings = data.get("ratings", data if isinstance(data, list) else [])
    if not isinstance(ratings, list):
        raise ValueError("Response JSON does not contain a ratings list")

    by_id = {}
    by_name = {}
    for r in ratings:
        if not isinstance(r, dict):
            continue
        sid = str(r.get("stimulus_id", "")).strip()
        name = str(r.get("file_name", "")).strip()
        if sid:
            by_id[sid] = r
        if name:
            by_name[name] = r

    rows = []
    for item in items:
        r = by_id.get(item.stimulus_id) or by_name.get(item.file_name)
        if r is None:
            rows.append({
                "dataset": dataset,
                "stimulus_id": item.stimulus_id,
                "file_name": item.file_name,
                "source_path": str(item.path),
                "llm": model,
                "valence": "",
                "arousal": "",
                "success": False,
                "error": "missing rating in model response",
            })
            continue
        try:
            valence = coerce_rating(r["valence"])
            arousal = coerce_rating(r["arousal"])
            rows.append({
                "dataset": dataset,
                "stimulus_id": item.stimulus_id,
                "file_name": item.file_name,
                "source_path": str(item.path),
                "llm": model,
                "valence": f"{valence:.2f}",
                "arousal": f"{arousal:.2f}",
                "success": True,
                "error": "",
            })
        except Exception as exc:
            rows.append({
                "dataset": dataset,
                "stimulus_id": item.stimulus_id,
                "file_name": item.file_name,
                "source_path": str(item.path),
                "llm": model,
                "valence": "",
                "arousal": "",
                "success": False,
                "error": f"invalid rating fields: {exc}",
            })
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "stimulus_id", "file_name", "source_path", "llm", "valence", "arousal", "success", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def chunks(items: list[ImageItem], size: int) -> Iterable[tuple[int, list[ImageItem]]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield (start // size + 1, items[start:start + size])


def call_with_retry(client: Any, kwargs: dict[str, Any], max_retries: int) -> str:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                if "max_tokens" in str(exc) and "max_completion_tokens" in str(exc):
                    fallback = dict(kwargs)
                    fallback["max_completion_tokens"] = fallback.pop("max_tokens")
                    response = client.chat.completions.create(**fallback)
                else:
                    raise
            text = response.choices[0].message.content
            if not text:
                raise ValueError("empty model response")
            return text
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            wait = min(2 ** attempt, 60)
            time.sleep(wait)
    raise RuntimeError(f"API call failed after {max_retries + 1} attempts: {last_exc}") from last_exc


def parse_dataset_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("dataset must be NAME=/path/to/images")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("dataset name cannot be empty")
    return name, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rate still-image datasets for valence/arousal with one request per dataset.")
    ap.add_argument("--dataset", action="append", type=parse_dataset_arg, help="Dataset as NAME=/path/to/images. Defaults to NAPS/IAPS/OASIS HPG paths.")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4"))
    ap.add_argument("--llm", default=os.environ.get("LLM", "gpt-5.4"), help="Label written to CSV.")
    ap.add_argument("--output-dir", default="output/image_va_ratings")
    ap.add_argument("--image-detail", choices=["low", "high"], default="low")
    ap.add_argument("--max-image-side", type=int, default=512)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128000)
    ap.add_argument("--batch-size", type=int, default=50,
                    help="Images per API request. NaviGator gpt-5.4 currently allows at most 50.")
    ap.add_argument("--dry-run", action="store_true", help="List images and estimate payload preparation only; do not call the model.")
    args = ap.parse_args()
    if args.batch_size > 50:
        raise ValueError("NaviGator gpt-5.4 allows at most 50 images per request; use --batch-size 50 or lower.")

    datasets = dict(args.dataset) if args.dataset else DEFAULT_DATASETS
    output_dir = Path(args.output_dir)

    client = None
    if not args.dry_run:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )

    combined_rows: list[dict[str, Any]] = []
    for dataset, directory in datasets.items():
        items = list_images(dataset, directory)
        if not items:
            raise FileNotFoundError(f"No images found for {dataset}: {directory}")

        print(f"{dataset}: {len(items)} images from {directory}")
        out_csv = output_dir / f"{dataset}_{args.llm}_va_ratings.csv"
        raw_json = output_dir / f"{dataset}_{args.llm}_raw_response.json"

        if args.dry_run:
            rows = [{
                "dataset": dataset,
                "stimulus_id": item.stimulus_id,
                "file_name": item.file_name,
                "source_path": str(item.path),
                "llm": args.llm,
                "valence": "",
                "arousal": "",
                "success": "",
                "error": "dry run",
            } for item in items]
            write_csv(out_csv, rows)
            combined_rows.extend(rows)
            print(f"  dry-run manifest written: {out_csv}")
            continue

        rows: list[dict[str, Any]] = []
        raw_json.parent.mkdir(parents=True, exist_ok=True)
        n_chunks = (len(items) + args.batch_size - 1) // args.batch_size
        for chunk_idx, chunk_items in chunks(items, args.batch_size):
            print(f"  request {chunk_idx}/{n_chunks}: {len(chunk_items)} images")
            prompt = build_prompt(dataset, chunk_items)
            content = make_content(
                chunk_items,
                prompt=prompt,
                image_detail=args.image_detail,
                max_side=args.max_image_side,
                jpeg_quality=args.jpeg_quality,
            )
            kwargs = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": content},
                ],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "response_format": {"type": "json_object"},
            }
            text = call_with_retry(client, kwargs, max_retries=args.max_retries)
            raw_chunk = output_dir / f"{dataset}_{args.llm}_raw_response_part{chunk_idx:03d}.json"
            raw_chunk.write_text(text, encoding="utf-8")

            rows.extend(normalize_rows(dataset, chunk_items, text, model=args.llm))
            write_csv(out_csv, rows)

        write_csv(out_csv, rows)
        combined_rows.extend(rows)
        ok = sum(1 for r in rows if r["success"] is True)
        print(f"  ratings written: {out_csv} ({ok}/{len(rows)} successful)")

    combined_csv = output_dir / f"all_datasets_{args.llm}_va_ratings.csv"
    write_csv(combined_csv, combined_rows)
    print(f"Combined CSV: {combined_csv}")


if __name__ == "__main__":
    main()
