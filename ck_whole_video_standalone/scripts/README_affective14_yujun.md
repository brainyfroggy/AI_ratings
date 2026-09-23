# CK Whole-Video Affective14 Yujun Runner

This runner rates each CK video on the 14 Cowen-Keltner affective dimensions
only:

`approach`, `arousal`, `attention`, `certainty`, `commitment`, `control`,
`dominance`, `effort`, `fairness`, `identity`, `obstruction`, `safety`,
`upswing`, `valence`.

The scale is 1.00-9.00, matching the paper's nine-point dimensional judgments
and the CK ground-truth CSV columns.

The runner sends one OpenAI request per video. For each video it first uses the
validated CK whole-video frame-selection method to extract frames from the
original `.mp4` file, then saves those frames under:

`output/ck/frames_yujun/<video_id>/frame_*.jpg`

The API request reads all generated frame files for that video directly. It does
not resize, re-encode, cap, or downsample them during the OpenAI request. Each
video is evaluated in one request containing all frames extracted for that same
video.

By default, the Python runner looks for videos in:

`N:\Experimental_Data\yujunchen\projects\data\original_ckvideo_data\ckvideo`

You can override this with `VIDEO_DIR=...` or `CK_VIDEOS_DIR=...` when using
`run_local_yujun.sh`.

Run from the `ck_whole_video_standalone` folder through the local wrapper.

Example test command, which makes one API request:

```bash
TEST_MODE=1 ./run_local_yujun.sh
```

Example full command:

```bash
./run_local_yujun.sh
```

Optional overrides:

```bash
CONCURRENCY=8 RPM=240 ./run_local_yujun.sh
./run_local_yujun.sh 0001 0002 0003
VIDEO_DIR="/path/to/ckvideo" ./run_local_yujun.sh
FRAMES_DIR="output/ck/frames_yujun" ./run_local_yujun.sh
OVERWRITE_FRAMES=1 ./run_local_yujun.sh
```

Output:

`output/ck/ratings/gpt-5.4_affective14_yujun.csv`
