---
title: Prompt Image Segmentation
emoji: 🎨
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
python_version : "3.14"
---

# Prompt Image Segmentation

Prompt-driven image segmentation. Give it an image and any text prompt - get back a binary mask, the cropped region, and a score telling you whether to trust them.

*Powered by GroundingDINO (open-set detection) + SAM 2.1 Hiera-Tiny (mask refinement), with YuNet and two small classifiers behind the optional face analysis - all permissively licensed. Effect-style architecture: explicit capabilities, typed recoverable failures, one composition root.*

## Example

| Original Image                         | Output Mask                    | Cropped Image                        | Prompt              |
| -------------------------------------- | ------------------------------ | ------------------------------------ | ------------------- |
| ![Original](examples/dog-original.png) | ![Mask](examples/dog-mask.png) | ![Cropped](examples/dog-cropped.png) | dog.                |
| ![Original](examples/cat-original.png) | ![Mask](examples/cat-mask.png) | ![Cropped](examples/cat-cropped.png) | cat.                |
| ![Original](examples/man-original.png) | ![Mask](examples/man-mask.png) | ![Cropped](examples/man-cropped.png) | costume. glasses.   |

## Compatibility

| Backend | How to run                                     |
| ------- | ---------------------------------------------- |
| CUDA    | `docker compose --profile cuda up --build`     |
| ROCm    | `docker compose --profile rocm up --build`     |
| CPU     | `docker compose --profile cpu up --build`      |
| Metal   | natively, see [Development](#development)      |

Metal has no Docker profile because Docker Desktop for macOS exposes no GPU to containers. Precision is picked per backend: `bfloat16` on CUDA/ROCm where the hardware supports it, `float16` on Metal, `float32` on CPU.

## Quickstart

### Production

```bash
cp .env.template .env

docker compose --profile cuda up --build
# API:           http://localhost:7860
# Documentation: http://localhost:7860/docs
```

The first start downloads ~700 MB of weights. `/readyz` answers 503 until they are resident, which is what the compose health check waits on.

### Development

```bash
poetry install
ENABLE_UI=true poetry run uvicorn app.interfaces.http:create_app \
    --factory --host 0.0.0.0 --port 7860
```

The Gradio UI is off by default: production serves the JSON API only and never imports `gradio`.

## API

Every route but `/healthz` and `/readyz` requires HTTP Basic credentials and is rate limited per client address.

### `GET /healthz`

Liveness. Answers as soon as the process is up, before the models finish loading.

```json
{ "status": "ok" }
```

### `GET /readyz`

Readiness. `503` until every model is resident, so a container runtime can tell a cold start from a wedged process.

```json
{ "status": "ready" }
```

### `GET /segmenters`

```json
{ "available": ["sam_dino"] }
```

### `POST /segment` (multipart form-data)

#### Request

| field         | type    | required | notes                                                |
|---------------|---------|----------|------------------------------------------------------|
| `image`       | file    | yes      | Any format Pillow can open, up to `MAX_UPLOAD_BYTES`.|
| `prompt`      | string  | yes      | `"cat.dog"` or `"cat. dog."` - separators tolerated. |
| `person_mode` | bool    | no       | Default `false`.                                     |
| `segmenter`   | string  | no       | Backend name. See `GET /segmenters`.                 |

#### Response

```json
{
  "prompt": "dog",
  "mask":  "<base64 PNG grayscale, cropped>",
  "image": "<base64 PNG original, cropped>",
  "bbox":  { "x": 120, "y": 340, "width": 512, "height": 780 },
  "detections": [
    { "detection_score": 0.95, "mask_score": 0.99, "confidence": 0.94 }
  ],
  "confidence": 0.94,
  "reliable": true,
  "segmenter": "sam_dino"
}
```

- `bbox` is the padded bounding box of the mask in the **original** image's coordinate system.
- `mask` and `image` are already cropped to `bbox` - align them by drawing at `(bbox.x, bbox.y)`.
- `person_mode=true` adds a `person` object:

```json
{ "genders": [0], "age_bands": ["sixties"], "is_adult": true }
```

`is_adult` is **fail-safe**, not merely likely: the age estimator classifies into ranges, and the band from ten to nineteen straddles the threshold, so it never certifies adulthood. `age_bands` is exposed so a caller can apply its own policy.

## Commands

| Command                              | Description                        |
| ------------------------------------ | ---------------------------------- |
| `poetry run poe lint`                | mypy (strict) + pylint             |
| `poetry run pyright`                 | Second-opinion type check          |
| `poetry run poe format`              | black                              |
| `poetry run poe test`                | Full suite                         |
| `poetry run pytest -m "not slow"`    | Fast suite, no model loading       |
| `poetry run poe audit`               | `pip-audit`                        |
| `poetry run python scripts/bench.py` | VRAM, latency and startup report   |

`scripts/bench.py --output bench/run.json` writes a comparable report; `memory.nvidia_smi_bytes` is the figure the sub-gigabyte budget is measured against.
