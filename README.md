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

| Original Image                           | Output Mask                      | Cropped Image                          | Prompt              |
| ---------------------------------------- | -------------------------------- | -------------------------------------- | ------------------- |
| ![Original](examples/dog-original.png)   | ![Mask](examples/dog-mask.png)   | ![Cropped](examples/dog-cropped.png)   | dog.                |
| ![Original](examples/cat-original.png)   | ![Mask](examples/cat-mask.png)   | ![Cropped](examples/cat-cropped.png)   | cat.                |
| ![Original](examples/human-original.png) | ![Mask](examples/human-mask.png) | ![Cropped](examples/human-cropped.png) | human.              |

## Compatibility

| Backend     | Linux  | Windows     | macOS  |
| ----------- | ------ | ----------- | ------ |
| CPU         | ✅     | ✅          | ✅     |
| NVIDIA CUDA | ✅     | ✅ via WSL2 | ❌     |
| AMD ROCm    | ✅     | ✅ via WSL2 | ❌     |

## Quickstart

### Production

```bash
cp .env.template .env

docker compose --profile [cuda|rocm|cpu] up --build -d
# API: http://localhost:7860
# Documentation: http://localhost:7860/docs
```

### Hugging Face Space

The Space builds this `Dockerfile`, so it needs three entries under
*Settings -> Variables and secrets*:
- `AUTH_MODE=none` for a public demo,
- `ENABLE_UI=true` to serve the Gradio page at `/`,
- `JOB_BACKEND=memory`, since the Space runs one container with no Redis
  beside it. Without it the `/jobs` routes answer `503`.

### Development

```bash
cp .env.template .env

poetry install --extras ui
poetry run uvicorn app.interfaces.http:create_app \
    --factory \
    --reload
```

## API

The `/jobs` routes require HTTP Basic credentials and are rate limited per client address. The three read-only routes below them are open: they expose no image data, and a probe has to reach them before any secret is configured.

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

### `POST /jobs` (multipart form-data)

#### Request

| field                 | type   | required | notes                                                        |
| --------------------- | ------ | -------- |------------------------------------------------------------- |
| `image`               | file   | yes      | Any format Pillow can open, up to `MAX_UPLOAD_BYTES`.        |
| `prompt`              | string | yes      | `"cat.dog"` or `"cat. dog."` - separators tolerated.         |
| `person_mode`         | bool   | no       | Default `false`.                                             |
| `segmenter`           | string | no       | Backend name. See `GET /segmenters`.                         |
| `minimum_confidence`  | float  | no       | Drop detections below this. Defaults to `MINIMUM_CONFIDENCE`.|
| `split_masks`         | bool   | no       | One mask per detection instead of one union. Default `false`.|
| `crop`                | bool   | no       | Crop to the subject. Default `true`.                         |
| `dilation_percentage` | float  | no       | Grow the mask. Defaults to `DILATION_PERCENTAGE`.            |
| `padding_percentage`  | float  | no       | Crop margin. Defaults to `MASK_PADDING_PERCENTAGE`.          |

#### Response

`202 Accepted` with an identifier to poll. Segmentation runs on a single accelerator, so the work is queued rather than held on an open connection:

```json
{
  "identifier": "0d9a1c1e-...",
  "state": "queued",
  "queue_position": 3,
  "created_at": 1755600000.0,
  "updated_at": 1755600000.0
}
```

### `GET /jobs/{identifier}`

The same body, with `state` moving to `running`, then to one of `succeeded`, `failed` or `cancelled`. A succeeded job carries `result`:

One region when the masks were merged, one per retained detection when `split_masks=true`. `image` is null when `crop=false`, since an uncropped image is the one the caller already holds.

```json
{
  "prompt": "dog",
  "regions": [
    {
      "bbox":  { "x": 120, "y": 340, "width": 512, "height": 780 },
      "mask":  "<base64 PNG grayscale>",
      "image": "<base64 PNG original, cropped>",
      "detection": {
        "detection_score": 0.95, "mask_score": 0.99, "confidence": 0.94
      }
    }
  ],
  "detections": [
    { "detection_score": 0.95, "mask_score": 0.99, "confidence": 0.94 }
  ],
  "confidence": 0.94,
  "reliable": true,
  "segmenter": "sam_dino"
}
```

With `person_mode=true` the same body carries one extra field:

### `GET /jobs/{identifier}/events` (WebSocket)

The same bodies, pushed as the job moves, closing on a terminal state. Carries the Basic credentials on the upgrade request, so it suits server-to-server callers; polling remains for anything that cannot hold a connection.

```bash
websocat -H "Authorization: Basic $(printf 'user:pass' | base64)" \
    ws://localhost:7860/jobs/<identifier>/events
```

### Webhooks

Pass `callback_url` to `POST /jobs` and the outcome is delivered there once, signed:

```http
X-Signature: sha256=<hmac of "<timestamp>." + body, keyed by WEBHOOK_SIGNING_SECRET>
X-Timestamp: 1755600000
```

Verify by recomputing the HMAC over the same material - the timestamp is inside it, so a captured delivery cannot be replayed under a fresh header. Deliveries go only to `https` addresses resolving to a public host - re-checked at delivery, since a name the caller controls can start answering with a loopback address while the job is queued - are never redirected, and are retried a bounded number of times. Without `WEBHOOK_SIGNING_SECRET` set, a `callback_url` is refused rather than sent unsigned; set, it must be at least 32 characters.

### `DELETE /jobs/{identifier}`

Withdraws a job that has not started. A running one answers `409`: the accelerator is already busy with it.

### Face analysis

```json
{
  "person": {
    "genders": [0],
    "age_bands": ["sixties"],
    "age_bands_digits": [[60, 69]],
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
