---
title: Prompt Image Segmentation
emoji: 🎨
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
python_version : "3.13"
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
| CUDA    | `docker compose --profile cuda up`             |
| ROCm    | `docker compose --profile rocm up`             |
| CPU     | `docker compose --profile cpu up`              |
| Metal   | natively, see [Development](#development)      |

## Quickstart

### Production

```bash
cp .env.template .env

docker compose --profile cuda up
# API: http://localhost:7860
# Documentation: http://localhost:7860/docs
```

The published images are pulled by the commands above. To fetch one
directly:

```bash
docker pull ghcr.io/maximedrn/prompt-image-segmentation:[cuda|rocm|cpu]
```

### Hugging Face Space

The Space builds this `Dockerfile`, so it needs three entries under *Settings -> Variables and secrets*:
- `AUTH_MODE=none` for a public demo,
- `ENABLE_UI=true` to serve the Gradio page at `/`,
- `INSTALL_UI=true` so the image is built with the optional extra.

### Development

```bash
poetry install --extras ui
ENABLE_UI=true poetry run uvicorn app.interfaces.http:create_app \
    --factory --host 0.0.0.0 --port 7860
```

## API

`POST /segment` requires HTTP Basic credentials and is rate limited per client address. The three routes below it are open: they expose no image data and a probe has to reach them before any secret is configured.

### `GET /healthz`

Liveness. Answers as soon as the process is up, before the models finish loading.

```json
{
  "status": "ok"
}
```

### `GET /readyz`

Readiness. `503` until every model is resident, so a container runtime can tell a cold start from a wedged process.

```json
{
  "status": "ready"
}
```

### `GET /segmenters`

```json
{
  "available": ["sam_dino"]
}
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

With `person_mode=true` the same body carries one extra field:

```json
{
  "person": {
    "genders": [0],
    "age_bands": ["sixties"],
    "is_adult": true
  }
}
```

## Commands

| Command                              | Description                        |
| ------------------------------------ | ---------------------------------- |
| `poetry run poe lint`                | mypy (strict) + pylint             |
| `poetry run pyright`                 | Second-opinion type check          |
| `poetry run poe format`              | black                              |
| `poetry run poe test`                | Full suite                         |
| `poetry run pytest -m "not slow"`    | Fast suite, no model loading       |
| `poetry run poe audit`               | `pip-audit`                        |
