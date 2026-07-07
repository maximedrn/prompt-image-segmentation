---
title: Prompt SAM Segmentation
emoji: 🎨
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
python_version : "3.13"
---

# Prompt SAM Segmentation

Prompt-driven image segmentation. Give it an image and any text prompt - get back a binary mask and the cropped region.

*Powered by GroundingDINO (open-set detection) + SAM ViT-H (mask refinement). Layered architecture; pluggable segmenter backends via a factory registry.*

## Example

| Original Image                         | Output Mask                    | Cropped Image                        | Prompt              |
| -------------------------------------- | ------------------------------ | ------------------------------------ | ------------------- |
| ![Original](examples/dog-original.png) | ![Mask](examples/dog-mask.png) | ![Cropped](examples/dog-cropped.png) | dog.                |
| ![Original](examples/cat-original.png) | ![Mask](examples/cat-mask.png) | ![Cropped](examples/cat-cropped.png) | cat.                |
| ![Original](examples/man-original.png) | ![Mask](examples/man-mask.png) | ![Cropped](examples/man-cropped.png) | costume. glasses.   |

## Quickstart

### Production

```bash
cp .env.template .env
docker compose -f docker/docker-compose.yaml up --build
# UI: http://localhost:7860
# Documentation: http://localhost:7860/docs
```

### Development

```bash
poetry install
poetry install --with person   # For human analysis.
poetry run uvicorn app.api:app --host 0.0.0.0 --port 7860
```

## API

### `GET /healthz`

Returns a simple health check.

#### Response

```json
{
  "status": "ok"
}
```

### `GET /segmenters`

Returns a list of available segmenter backends.

#### Response

```json
{
  "available":[
    "sam_dino"
  ]
}
```

### `POST /segment` (multipart form-data)

#### Request

| field         | type    | required | notes                                                |
|---------------|---------|----------|------------------------------------------------------|
| `image`       | file    | yes      | Any format Pillow can open.                          |
| `prompt`      | string  | yes      | `"cat.dog"` or `"cat. dog."` - separators tolerated. |
| `person_mode` | bool    | no       | Default `false`.                                     |
| `segmenter`   | string  | no       | Backend name. See `GET /segmenters`.                 |

#### Response - generic mode (`person_mode=false`)

```json
{
  "prompt": "dog",
  "mask":  "<base64 PNG grayscale, cropped>",
  "image": "<base64 PNG original, cropped>",
  "bbox":  { "x": 120, "y": 340, "width": 512, "height": 780 },
  "detections": 3,
  "segmenter": "sam_dino"
}
```

- `bbox` is the padded bounding box of the mask in the **original** image's coordinate system.
- `mask` and `image` are already cropped to `bbox` - align them by drawing at `(bbox.x, bbox.y)`.

#### Response - person mode (`person_mode=true`)

```json
{
  "prompt": "dog",
  "mask":  "<base64 PNG grayscale, cropped>",
  "image": "<base64 PNG original, cropped>",
  "bbox":  { "x": 120, "y": 340, "width": 512, "height": 780 },
  "detections": 3,
  "segmenter": "sam_dino",
  "person": {
    "genders": [0, 1],
    "is_adult": true
  }
}
```

- `genders`: `0` = Male, `1` = Female, one per detected face.
- `is_adult`: `true` iff every face is ≥ 18 years old.

### Errors

```json
{ "error": "no_detection", "message": "..." } // 422
{ "error": "invalid_prompt", "message": "..." } // 422
{ "error": "unknown_backend", "message": "..." } // 400
{ "error": "internal", "message": "..." } // 500
```
