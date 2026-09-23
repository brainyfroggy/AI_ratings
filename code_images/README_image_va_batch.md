# IAPS/NAPS/OASIS OpenAI Image Rating Test

This script rates IAPS, NAPS, or OASIS pictures with a CK-derived
valence/arousal prompt using OpenAI Responses.

Dataset directories:

- `iaps`: `N:\Experimental_Data\yujunchen\projects\data\IAPS1182\IAPS1182`
- `naps`: `N:\Experimental_Data\yujunchen\projects\data\NAPS_H\NAPS_H`
- `oasis`: `N:\Experimental_Data\yujunchen\projects\data\OASIS\images`

Default request size is `50` images/request. A normal run processes all three
datasets by default, in chunks of 50, until all images have ratings. This is
conservative because the OpenAI API prompt/input window is about 200 MB, and
the largest raw images here are from NAPS at about 2.5 MB each.

Images are sent as original file bytes. The script does not resize, downsample,
or JPEG-recompress them.

The image filename is preserved as the model `image_id` and is saved in the CSV
as `image_name`.

## Dry Run

```powershell
python .\code_images\openai_image_va_batch.py --dry-run
python .\code_images\openai_image_va_batch.py --dataset iaps --dry-run
python .\code_images\openai_image_va_batch.py --dataset naps --dry-run
python .\code_images\openai_image_va_batch.py --dataset oasis --dry-run
```

This verifies image discovery, compression, and request construction without
calling the API.

## Test Mode

Run exactly one request per dataset:

```powershell
python .\code_images\openai_image_va_batch.py --test-mode
```

By default this sends `50` images/request for `iaps`, `naps`, and `oasis`.
To test a different one-request size:

```powershell
python .\code_images\openai_image_va_batch.py --test-mode --batch-sizes 25
```

Dry-run the same workflow without API calls:

```powershell
python .\code_images\openai_image_va_batch.py --test-mode --dry-run
```

## Full Dataset Run

```powershell
$env:OPENAI_API_KEY = "..."
python .\code_images\openai_image_va_batch.py
```

This processes all images in `iaps`, `naps`, and `oasis`, 50 images per request
by default. To process only one dataset:

```powershell
python .\code_images\openai_image_va_batch.py --dataset iaps
```

To change images per request:

```powershell
python .\code_images\openai_image_va_batch.py --batch-sizes 25
```

Optional Navigator/proxy settings:

```powershell
$env:OPENAI_BASE_URL = "..."
$env:OPENAI_MODEL = "gpt-5.4"
python .\code_images\openai_image_va_batch.py
```

The script follows `N:\Experimental_Data\SimonaS\valence_arousal\va_batch.py`
for API connection: it constructs `OpenAI()` directly, so the SDK reads
`OPENAI_API_KEY` and optional `OPENAI_BASE_URL` from your environment.

## Batch API Mode

Use this if you want the same `/v1/responses` request body submitted through
OpenAI Batch API:

```powershell
python .\code_images\openai_image_va_batch.py --dataset iaps --mode batch
```

## Outputs

Outputs go to:

```text
code_images/image_va_batch_results/<dataset>/
```

Important files:

- `summary.json`: pass/fail status, request payload size, and chunk progress
- `ratings_all.csv`: cumulative ratings for the dataset; includes `image_name`
- `ratings_<dataset>_chunk_*.csv`: per-request chunk ratings
- `responses_raw_<dataset>_chunk_*.json` or `batch_output_<dataset>_chunk_*.jsonl`: raw API output
- `error_<N>.txt`: traceback for any failed attempt

The prompt text is also copied into `ck_image_va_prompt.md` for review and reuse
in the full IAPS/NAPS/OASIS pipeline.

## SSL Certificate Error

If `OpenAI()` fails before any request with `FileNotFoundError` from
`SSL_CERT_FILE`, the script now repairs stale certificate environment variables
inside the current Python process. In Git Bash you can also fix the shell
manually before running:

```bash
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
```
