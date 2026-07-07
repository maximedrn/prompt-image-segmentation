"""Route handlers grouped by concern (meta + segmentation)."""

from io import BytesIO

from PIL import Image as PilImage
from PIL.Image import Image
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.api.auth import require_basic_auth
from app.api.schemas import ErrorSchema, HealthSchema
from app.domain import SegmentResponse
from app.services import SegmentationService

meta_router: APIRouter = APIRouter(tags=["meta"])
segmentation_router: APIRouter = APIRouter(
    tags=["segmentation"],
    dependencies=[Depends(require_basic_auth)],
)


@meta_router.get("/healthz", response_model=HealthSchema)
def healthcheck() -> HealthSchema:
    """Return ``{"status": "ok"}``. No auth required.

    :returns: A HealthSchema with ``status == "ok"``.
    :rtype: app.api.schemas.HealthSchema
    """
    return HealthSchema(status="ok")


@meta_router.get("/segmenters")
def list_segmenters() -> dict[str, list[str]]:
    """Enumerate registered segmentation backends by name.

    :returns: ``{"available": [...backend names...]}``.
    :rtype: dict[str, list[str]]
    """
    return {"available": SegmentationService().available_backends()}


def _read_image(upload: UploadFile) -> Image:
    """Decode ``upload`` into an RGB PIL image.

    :param upload: Multipart file upload from FastAPI.
    :type upload: fastapi.UploadFile
    :returns: A PIL image in RGB mode.
    :rtype: PIL.Image.Image
    :raises fastapi.HTTPException: 422 on empty upload or decode error.
    """
    if not (payload := upload.file.read()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty upload.",
        )
    try:
        return PilImage.open(BytesIO(payload)).convert("RGB")
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid image: {error}",
        ) from error


@segmentation_router.post(
    "/segment",
    response_model=SegmentResponse,
    responses={
        422: {"model": ErrorSchema},
        400: {"model": ErrorSchema},
        500: {"model": ErrorSchema},
    },
)
def segment_endpoint(
    image: UploadFile = File(..., description="Image to segment."),
    prompt: str = Form(
        ...,
        description="Text prompt (comma or dot separated).",
    ),
    person_mode: bool = Form(
        default=False,
        description="Enable face detection (gender/age).",
    ),
    segmenter: str | None = Form(
        default=None,
        description="Backend name (default: sam_dino).",
    ),
) -> SegmentResponse:
    """Run segmentation on ``image`` and return the base64 payload.

    :param image: Uploaded image file (multipart).
    :type image: fastapi.UploadFile
    :param prompt: GroundingDINO text prompt (dot/comma separated).
    :type prompt: str
    :param person_mode: Enable face detection.
    :type person_mode: bool
    :param segmenter: Optional backend override.
    :type segmenter: str | None
    :returns: The segmentation response.
    :rtype: app.domain.SegmentResponse
    """
    pil_image: Image = _read_image(image)
    return SegmentationService().run(
        image=pil_image,
        prompt=prompt,
        person_mode=person_mode,
        segmenter_name=segmenter,
    )


__all__: list[str] = [
    "meta_router",
    "segmentation_router",
    "healthcheck",
    "list_segmenters",
    "segment_endpoint",
]
