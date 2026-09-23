#!/usr/bin/env python3
"""Rate Cowen-Keltner (CK) short video clips with an LLM — ONE WHOLE-VIDEO judgment per clip.

Unlike run_ck_rating.py (which samples many timestamps and peak-end-AVERAGES them into a
single value), this script gives the LLM the SAME task as the human raters: watch the whole
clip once and give ONE overall emotion rating. Mechanically: extract N key/uniform frames
spanning the entire clip, send them in a SINGLE prompt asking for the OVERALL emotional
content (prompt_mode="overall"), and record one rating per video.

CK clips have no audio, so this is frames-only.

Dimensions:
  --dimensions va        -> valence + arousal (1-9)                      [default, simplest]
  --dimensions ck_full   -> 34 emotion categories + 14 affective dims    [matches the full human GT]

Output: one row per video in <ratings-dir>/<llm>_ratings.csv
  va     : stimulus_id, llm, valence, arousal, success, error
  ck_full: stimulus_id, llm, <48 CK keys...>, success, error

Human ground truth to compare against: data/ground_truth/ck_ground_truth.csv (2185 rows x 48 dims).

Example (HPG, GPT-5.4 via NaviGator):
  python scripts/run_ck_whole_video_rating.py --llm gpt-5.4 --model gpt-5.4 \
      --dimensions va --image-detail low --temperature 0.4 --concurrency 8 --rpm 600 --resume
"""
from __future__ import annotations
import argparse
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ck_config import CKConfig
from src.ck_data_loader import CKDataLoader
from src.config import Config
from src.llm_rater import LLMRater
from src.rating_strategy import STRATEGY_CK, STRATEGY_CK_FULL
from src.ck_dimensions import CK_ALL_KEYS


def video_duration_ms(path: Path) -> float:
    """Whole-clip length in ms (so the single call's frames span [0, end])."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0 or n <= 0:
        return 5000.0  # fallback; rate_video_timestamp clamps to the real video length
    return (n / fps) * 1000.0


class RateLimiter:
    """Simple global RPM limiter for parallel mode."""
    def __init__(self, rpm: float):
        self.delay = 60.0 / rpm if rpm > 0 else 0.0
        self.lock = threading.Lock(); self.next_t = 0.0

    def acquire(self):
        if self.delay <= 0:
            return
        with self.lock:
            now = time.monotonic(); wait = self.next_t - now
            if wait > 0:
                time.sleep(wait)
            self.next_t = max(now, self.next_t) + self.delay


def rate_one_video(vid, ck_config, rater, llm, dims, limiter):
    """ONE whole-video call: frames spanning [0, end] + 'rate the OVERALL emotional content'."""
    path = ck_config.get_video_path(vid)
    dur = video_duration_ms(path)
    if limiter:
        limiter.acquire()
    r = rater.rate_video_timestamp(vid, int(dur), llm, video_path=path)  # prompt_mode="overall" set on the strategy
    row = {"stimulus_id": vid, "llm": llm, "success": bool(r.success), "error": r.error or ""}
    if dims == "ck_full":
        extra = r.extra_ratings or {}
        for k in CK_ALL_KEYS:
            row[k] = extra.get(k, float("nan"))
    else:
        row["valence"] = r.valence; row["arousal"] = r.arousal
    return row


def main():
    ap = argparse.ArgumentParser(description="CK WHOLE-VIDEO overall emotion rating (one rating per clip).")
    ap.add_argument("--llm", default="gpt-5.4", help="LLM name (gpt-5.4, gemini-3.1-pro, ...).")
    ap.add_argument("--model", default=None, help="Backend model id (defaults to --llm).")
    ap.add_argument("--dimensions", choices=["va", "ck_full"], default="va",
                    help="'va' (valence/arousal) or 'ck_full' (34 categories + 14 dims, matches human GT).")
    ap.add_argument("--image-detail", choices=["low", "high"], default="low")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--rpm", type=float, default=None, help="Requests/min cap for parallel mode.")
    ap.add_argument("--rate-limit", type=float, default=0.0, help="Per-call sleep (sequential mode).")
    ap.add_argument("--ratings-dir", default=None, help="Output dir (default: CKConfig.ratings_output_dir).")
    ap.add_argument("--videos", nargs="*", default=None, help="Subset of video ids (default: all 2185).")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true", help="Skip videos already in the output CSV.")
    args = ap.parse_args()

    ck = CKConfig(); ck.ensure_directories()
    if args.model:
        ck.openai_model = args.model
    base = Config()
    base.openai_model = args.model or args.llm
    base.rate_limit_delay = args.rate_limit

    # THE key setting: rate the WHOLE video's OVERALL emotion (not the current frame).
    strat = STRATEGY_CK_FULL if args.dimensions == "ck_full" else STRATEGY_CK
    strat = replace(strat, prompt_mode="overall", image_detail=args.image_detail, temperature=args.temperature)
    rater = LLMRater(base, strategy=strat)

    loader = CKDataLoader(ck)
    vids = [str(v) for v in (args.videos if args.videos else loader.get_video_ids())]

    out_dir = Path(args.ratings_dir) if args.ratings_dir else ck.ratings_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.llm}_ratings.csv"

    done = set()
    rows = []
    if args.resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(prev[prev.get("success", False) == True]["stimulus_id"].astype(str))
        print(f"Resume: {len(done)} videos already done; {len(vids)-len(done)} to go.")
    todo = [v for v in vids if v not in done]

    limiter = RateLimiter(args.rpm or (60.0 / args.rate_limit if args.rate_limit > 0 else 30)) if args.concurrency > 1 else None
    if limiter:
        base.rate_limit_delay = 0.0

    print(f"CK WHOLE-VIDEO rating | {args.llm} | dims={args.dimensions} | overall prompt | "
          f"{len(todo)}/{len(vids)} videos | concurrency {args.concurrency} -> {out_csv}")

    def flush():
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    if args.concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(rate_one_video, v, ck, rater, args.llm, args.dimensions, limiter): v for v in todo}
            for i, f in enumerate(tqdm(as_completed(futs), total=len(futs))):
                rows.append(f.result())
                if (i + 1) % args.save_every == 0:
                    flush()
    else:
        for i, v in enumerate(tqdm(todo)):
            rows.append(rate_one_video(v, ck, rater, args.llm, args.dimensions, None))
            if args.rate_limit > 0:
                time.sleep(args.rate_limit)
            if (i + 1) % args.save_every == 0:
                flush()
    flush()
    ok = sum(1 for r in rows if r.get("success"))
    print(f"Done. {ok}/{len(rows)} successful -> {out_csv}")


if __name__ == "__main__":
    main()
