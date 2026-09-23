"""Refresh ck_keyframe_distribution_summary.ipynb with executable matplotlib charts."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "ck_whole_video_keyframes" / "summary.json"
NB_PATH = ROOT / "code_videos" / "ck_keyframe_distribution_summary.ipynb"


def png_for(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def table_output(df: pd.DataFrame, execution_count: int) -> dict:
    return {
        "output_type": "execute_result",
        "execution_count": execution_count,
        "data": {
            "text/html": df.to_html(index=False),
            "text/plain": df.to_string(index=False),
        },
        "metadata": {},
    }


def display_png(png: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"image/png": png},
        "metadata": {},
    }


def bar_png(labels, values, title, xlabel, ylabel="Videos", color="#3B82F6"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    try:
        numeric_labels = [float(label) for label in labels]
    except (TypeError, ValueError):
        numeric_labels = []
    if numeric_labels and all(label.is_integer() for label in numeric_labels):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if len(labels) > 25:
        step = max(1, len(labels) // 20)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=45, ha="right")
    else:
        ax.tick_params(axis="x", rotation=0)
    return png_for(fig)


def dual_axis_png(df: pd.DataFrame) -> str:
    plot_df = df.sort_values(["video_num", "video_id"]).reset_index(drop=True)
    x = range(len(plot_df))
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.plot(x, plot_df["saved_frames"], color="#2563EB", linewidth=1.1, label="Extracted/saved frames")
    ax1.set_ylabel("Extracted/saved frames", color="#2563EB")
    ax1.set_ylim(0, 20)
    ax1.set_yticks(list(range(0, 21, 1)))
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(x, plot_df["duration_sec"], color="#F97316", linewidth=1.2, label="Video length (s)")
    ax2.set_ylabel("Video length (seconds)", color="#F97316")
    ax2.tick_params(axis="y", labelcolor="#F97316")

    tick_positions = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, len(plot_df) - 1]
    tick_positions = [i for i in tick_positions if 0 <= i < len(plot_df)]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(plot_df.loc[tick_positions, "video_id"].astype(str), rotation=45, ha="right")
    ax1.set_xlabel("Video ID order")
    ax1.set_title("Extracted Frames and Video Length by Video")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    fig.tight_layout()
    return png_for(fig)


def scatter_png(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(df["duration_sec"], df["saved_frames"], s=12, alpha=0.35, color="#2563EB")
    ax.set_title("Duration vs Extracted/Saved Frames")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Extracted/saved frames")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    return png_for(fig)


def set_chart_cell(nb: dict, heading_prefix: str, source: str, outputs: list[dict]) -> None:
    for idx, cell in enumerate(nb["cells"]):
        if "".join(cell.get("source", [])).startswith(heading_prefix):
            nb["cells"][idx + 1]["source"] = source.splitlines(keepends=True)
            nb["cells"][idx + 1]["outputs"] = outputs
            return
    raise RuntimeError(f"Heading not found: {heading_prefix}")


def set_chart_cell_any(
    nb: dict,
    heading_prefixes: list[str],
    new_heading: str | None,
    source: str,
    outputs: list[dict],
) -> None:
    for idx, cell in enumerate(nb["cells"]):
        if any("".join(cell.get("source", [])).startswith(prefix) for prefix in heading_prefixes):
            if new_heading is not None:
                nb["cells"][idx]["source"] = [new_heading]
            nb["cells"][idx + 1]["source"] = source.splitlines(keepends=True)
            nb["cells"][idx + 1]["outputs"] = outputs
            return
    raise RuntimeError(f"Heading not found: {heading_prefixes}")


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    df = pd.DataFrame([{k: v for k, v in row.items() if k != "frames"} for row in summary["results"]])
    df["duration_sec"] = df["duration_ms"] / 1000.0
    df["frame_shortfall"] = df["requested_frames"] - df["saved_frames"]
    df["video_num"] = pd.to_numeric(df["video_id"], errors="coerce")

    saved_counts = df["saved_frames"].value_counts().sort_index().rename_axis("extracted_saved_frames").reset_index(name="n_videos")
    requested_counts = df["requested_frames"].value_counts().sort_index().rename_axis("requested_frames").reset_index(name="n_videos")
    sampling_counts = df["effective_sampling"].value_counts().rename_axis("effective_sampling").reset_index(name="n_videos")
    shortfall_counts = df["frame_shortfall"].value_counts().sort_index().rename_axis("requested_minus_saved").reset_index(name="n_videos")

    max_bin = int(math.ceil(df["duration_sec"].max()))
    duration_counts_1s = (
        df.assign(duration_bin_start_sec=df["duration_sec"].apply(lambda x: int(math.floor(x))))
        ["duration_bin_start_sec"]
        .value_counts()
        .sort_index()
        .reindex(range(0, max_bin + 1), fill_value=0)
        .rename_axis("duration_bin_start_sec")
        .reset_index(name="n_videos")
    )
    duration_counts_1s["duration_bin_sec"] = duration_counts_1s["duration_bin_start_sec"].map(lambda x: f"{x}-{x+1}")
    duration_counts_1s = duration_counts_1s[["duration_bin_sec", "duration_bin_start_sec", "n_videos"]]

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Make sure the top data-loading cell imports matplotlib for reruns.
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("from pathlib import Path"):
            if "import matplotlib.pyplot as plt" not in source:
                source = source.replace("import pandas as pd\n", "import pandas as pd\nimport matplotlib.pyplot as plt\nfrom matplotlib.ticker import MaxNLocator\n")
            elif "from matplotlib.ticker import MaxNLocator" not in source:
                source = source.replace("import matplotlib.pyplot as plt\n", "import matplotlib.pyplot as plt\nfrom matplotlib.ticker import MaxNLocator\n")
            cell["source"] = source.splitlines(keepends=True)
            break

    set_chart_cell(
        nb,
        "## Extracted/Saved Frames Per Video",
        """saved_counts = df['saved_frames'].value_counts().sort_index().rename_axis('extracted_saved_frames').reset_index(name='n_videos')
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(saved_counts['extracted_saved_frames'], saved_counts['n_videos'], color='#3B82F6')
ax.set_title('Extracted/Saved Frames Per Video (Observed Min = 2)')
ax.set_xlabel('Extracted/saved frames')
ax.set_ylabel('Videos')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(axis='y', alpha=0.25)
plt.show()
saved_counts
""",
        [
            display_png(bar_png(saved_counts["extracted_saved_frames"], saved_counts["n_videos"], "Extracted/Saved Frames Per Video (Observed Min = 2)", "Extracted/saved frames")),
            table_output(saved_counts, 3),
        ],
    )

    set_chart_cell_any(
        nb,
        ["## CK Target Frames Before Read Failures", "## Requested Frame Budget"],
        "## CK Target Frames Before Read Failures\n\nThis is the CK sampling target/requested count before OpenCV frame-read failures. It is not the same as the number of files saved in each video folder.\n",
        """requested_counts = df['requested_frames'].value_counts().sort_index().rename_axis('requested_frames').reset_index(name='n_videos')
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(requested_counts['requested_frames'], requested_counts['n_videos'], color='#10B981')
ax.set_title('CK Target Frames Before Read Failures')
ax.set_xlabel('Target/requested frames before extraction')
ax.set_ylabel('Videos')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(axis='y', alpha=0.25)
plt.show()
requested_counts
""",
        [
            display_png(bar_png(requested_counts["requested_frames"], requested_counts["n_videos"], "CK Target Frames Before Read Failures", "Target/requested frames before extraction", color="#10B981")),
            table_output(requested_counts, 4),
        ],
    )

    set_chart_cell(
        nb,
        "## Duration Distribution (1-Second Bins)",
        """# 1-second duration bins: [0,1), [1,2), ...
import math

max_bin = int(math.ceil(df['duration_sec'].max()))
duration_counts_1s = (
    df.assign(duration_bin_start_sec=df['duration_sec'].apply(lambda x: int(math.floor(x))))
    ['duration_bin_start_sec']
    .value_counts()
    .sort_index()
    .reindex(range(0, max_bin + 1), fill_value=0)
    .rename_axis('duration_bin_start_sec')
    .reset_index(name='n_videos')
)
duration_counts_1s['duration_bin_sec'] = duration_counts_1s['duration_bin_start_sec'].map(lambda x: f'{x}-{x+1}')
duration_counts_1s = duration_counts_1s[['duration_bin_sec', 'duration_bin_start_sec', 'n_videos']]

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(duration_counts_1s['duration_bin_start_sec'], duration_counts_1s['n_videos'], color='#F59E0B')
ax.set_title('Video Duration Distribution (1-Second Bins; 0 = 0-1 sec)')
ax.set_xlabel('Duration bin start in seconds (0 means 0-1 sec)')
ax.set_ylabel('Videos')
ax.set_xticks(range(0, max_bin + 1, 5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(axis='y', alpha=0.25)
plt.show()
duration_counts_1s
""",
        [
            display_png(bar_png(duration_counts_1s["duration_bin_sec"], duration_counts_1s["n_videos"], "Video Duration Distribution (1-Second Bins; 0 = 0-1 sec)", "Duration bin start in seconds (0 means 0-1 sec)", color="#F59E0B")),
            table_output(duration_counts_1s, 5),
        ],
    )

    set_chart_cell(
        nb,
        "## Effective Sampling Mode",
        """sampling_counts = df['effective_sampling'].value_counts().rename_axis('effective_sampling').reset_index(name='n_videos')
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(sampling_counts['effective_sampling'], sampling_counts['n_videos'], color='#8B5CF6')
ax.set_title('Effective Sampling Mode')
ax.set_xlabel('Mode')
ax.set_ylabel('Videos')
ax.grid(axis='y', alpha=0.25)
plt.show()
sampling_counts
""",
        [
            display_png(bar_png(sampling_counts["effective_sampling"], sampling_counts["n_videos"], "Effective Sampling Mode", "Mode", color="#8B5CF6")),
            table_output(sampling_counts, 6),
        ],
    )

    set_chart_cell(
        nb,
        "## Frame Shortfall",
        """shortfall_counts = df['frame_shortfall'].value_counts().sort_index().rename_axis('requested_minus_saved').reset_index(name='n_videos')
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(shortfall_counts['requested_minus_saved'], shortfall_counts['n_videos'], color='#EF4444')
ax.set_title('Frame Shortfall')
ax.set_xlabel('Requested minus saved frames')
ax.set_ylabel('Videos')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(axis='y', alpha=0.25)
plt.show()
shortfall_counts
""",
        [
            display_png(bar_png(shortfall_counts["requested_minus_saved"], shortfall_counts["n_videos"], "Frame Shortfall", "Requested minus saved frames", color="#EF4444")),
            table_output(shortfall_counts, 7),
        ],
    )

    set_chart_cell(
        nb,
        "## Extracted Frames and Video Length by Video",
        """plot_df = df.sort_values([pd.to_numeric(df['video_id'], errors='coerce'), 'video_id']).reset_index(drop=True)
x = range(len(plot_df))
fig, ax1 = plt.subplots(figsize=(12, 4.5))
ax1.plot(x, plot_df['saved_frames'], color='#2563EB', linewidth=1.1, label='Extracted/saved frames')
ax1.set_ylabel('Extracted/saved frames', color='#2563EB')
ax1.set_ylim(0, 20)
ax1.set_yticks(range(0, 21, 1))
ax1.tick_params(axis='y', labelcolor='#2563EB')
ax1.grid(axis='y', alpha=0.2)

ax2 = ax1.twinx()
ax2.plot(x, plot_df['duration_sec'], color='#F97316', linewidth=1.2, label='Video length (s)')
ax2.set_ylabel('Video length (seconds)', color='#F97316')
ax2.tick_params(axis='y', labelcolor='#F97316')

tick_positions = [i for i in [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, len(plot_df) - 1] if 0 <= i < len(plot_df)]
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(plot_df.loc[tick_positions, 'video_id'].astype(str), rotation=45, ha='right')
ax1.set_xlabel('Video ID order')
ax1.set_title('Extracted Frames and Video Length by Video')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')
plt.show()
""",
        [display_png(dual_axis_png(df))],
    )

    set_chart_cell_any(
        nb,
        ["## Duration vs Extracted/Saved Frames", "## Duration vs Saved Frames"],
        "## Duration vs Extracted/Saved Frames\n",
        """fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(df['duration_sec'], df['saved_frames'], s=12, alpha=0.35, color='#2563EB')
ax.set_title('Duration vs Extracted/Saved Frames')
ax.set_xlabel('Duration (seconds)')
ax.set_ylabel('Extracted/saved frames')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(alpha=0.25)
plt.show()
""",
        [display_png(scatter_png(df))],
    )

    NB_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Updated {NB_PATH}")


if __name__ == "__main__":
    main()
