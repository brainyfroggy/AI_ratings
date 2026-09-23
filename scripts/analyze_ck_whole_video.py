#!/usr/bin/env python3
"""Compare LLM whole-video CK ratings to the human ground truth, across the 2185 clips.

The unit of analysis is the VIDEO (one LLM rating vs one human rating per clip) — NOT a
time course. We report between-video correlation (how well the LLM's overall rating of a
clip tracks the humans' overall rating) per dimension.

  LLM ratings : <ratings-dir>/<llm>_ratings.csv  (from run_ck_whole_video_rating.py)
  human GT    : data/ground_truth/ck_ground_truth.csv  (2185 x 48: 34 categories [0-1] + 14 dims [1-9])

Pearson is scale-invariant, so LLM-scale vs human-scale mismatches don't matter for r.
For 'va' we also report Spearman. For 'ck_full' we summarise per-dimension r and split
categories (proportions) vs affective dimensions.

Output:
  <ratings-dir>/<llm>_vs_human.csv
  doc/figures/ck_<llm>_whole_video.{pdf,png}
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
GT = ROOT / "data/ground_truth/ck_ground_truth.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True, help="LLM ratings CSV (e.g. .../gpt-5.4_ratings.csv)")
    ap.add_argument("--llm", default="llm")
    args = ap.parse_args()

    llm = pd.read_csv(args.ratings)
    llm = llm[llm.get("success", True) != False].copy()
    llm["stimulus_id"] = llm["stimulus_id"].astype(str)
    g = pd.read_csv(GT); g["stimulus_id"] = g["stimulus_id"].astype(str)

    # which dimensions are present in BOTH the LLM output and the human GT?
    dims = [c for c in g.columns if c != "stimulus_id" and c in llm.columns]
    m = pd.merge(llm[["stimulus_id"] + dims], g[["stimulus_id"] + dims],
                 on="stimulus_id", suffixes=("_llm", "_hum"))
    print(f"{args.llm}: {len(m)} videos matched, {len(dims)} dimensions compared.")

    rows = []
    for d in dims:
        x, y = m[f"{d}_llm"].to_numpy(float), m[f"{d}_hum"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            rows.append({"dim": d, "pearson": np.nan, "spearman": np.nan, "n": int(ok.sum())}); continue
        rows.append({"dim": d, "pearson": stats.pearsonr(x[ok], y[ok])[0],
                     "spearman": stats.spearmanr(x[ok], y[ok])[0], "n": int(ok.sum())})
    res = pd.DataFrame(rows)
    out_csv = Path(args.ratings).with_name(f"{args.llm}_vs_human.csv")
    res.to_csv(out_csv, index=False)

    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nMedian Pearson r across {len(res)} dims: {res.pearson.median():+.3f}")
    if {"valence", "arousal"}.issubset(set(res.dim)):
        rv = res[res.dim == "valence"].iloc[0]; ra = res[res.dim == "arousal"].iloc[0]
        print(f"  valence r {rv.pearson:+.3f}  | arousal r {ra.pearson:+.3f}")

    # figure: per-dimension Pearson r (sorted)
    rs = res.dropna(subset=["pearson"]).sort_values("pearson")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(rs))))
    ax.barh(np.arange(len(rs)), rs.pearson, color="#4c72b0")
    ax.axvline(0, color="k", lw=0.6); ax.set_yticks(np.arange(len(rs))); ax.set_yticklabels(rs.dim, fontsize=7)
    ax.set_xlabel("Pearson r (LLM vs human, across videos)"); ax.set_xlim(-0.3, 1.0)
    ax.set_title(f"CK whole-video: {args.llm} vs human ({len(m)} clips)\nmedian r {res.pearson.median():+.2f}")
    fig.tight_layout()
    FIG = ROOT / "doc/figures"; FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"ck_{args.llm}_whole_video.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"ck_{args.llm}_whole_video.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_csv}\nSaved {FIG/('ck_'+args.llm+'_whole_video.pdf')}")

    # figure: valence & arousal scatter (one dot per clip, LLM vs human)
    if {"valence", "arousal"}.issubset(set(res.dim)):
        figs, axs = plt.subplots(1, 2, figsize=(9, 4.3))
        for ax, d, col in zip(axs, ["valence", "arousal"], ["#4c72b0", "#c44e52"]):
            x, y = m[f"{d}_llm"].to_numpy(float), m[f"{d}_hum"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
            r = res[res.dim == d].iloc[0].pearson
            if len(x) == 0:
                ax.set_title(f"{d}: no data"); ax.set_xlabel(f"{args.llm} {d}")
                ax.set_ylabel(f"human {d}"); continue
            ax.scatter(x, y, s=10, alpha=0.35, color=col, edgecolors="none")
            lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.6)  # identity
            if len(x) >= 2 and np.std(x) > 0:  # least-squares fit
                b, a = np.polyfit(x, y, 1)
                xs = np.array([lo, hi]); ax.plot(xs, b * xs + a, color=col, lw=1.5)
            ax.set_xlabel(f"{args.llm} {d}"); ax.set_ylabel(f"human {d}")
            ax.set_title(f"{d}: r = {r:+.2f} (n = {len(x)})")
        figs.suptitle(f"CK whole-video: {args.llm} vs human, per clip", fontsize=12)
        figs.tight_layout(rect=[0, 0, 1, 0.96])
        figs.savefig(FIG / f"ck_{args.llm}_va_scatter.pdf", bbox_inches="tight")
        figs.savefig(FIG / f"ck_{args.llm}_va_scatter.png", dpi=150, bbox_inches="tight")
        print(f"Saved {FIG/('ck_'+args.llm+'_va_scatter.pdf')}")


if __name__ == "__main__":
    main()
