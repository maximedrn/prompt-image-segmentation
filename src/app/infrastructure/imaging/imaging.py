"""Pixel plumbing: bytes to array, array to wire, mask dilation."""

from base64 import b64encode
from io import BytesIO
from typing import final

from cv2 import dilate as cv2_dilate
from numpy import array, ones, uint8
from numpy.typing import NDArray
from PIL import Image as PilImage
from PIL.Image import Image

from app.domain import (
    ImageDecodeFailed,
    ImageFormat,
    ImageMode,
    MaskArray,
    PercentageBounds,
    SourceImage,
    clamp_percentage,
)
from app.infrastructure.imaging.constants import Dilation
from app.infrastructure.imaging.types import TextEncoding


def decode_image(payload: bytes) -> SourceImage:
    """Decode uploaded bytes into an RGB source image.

    :param payload: Raw bytes as received from the client.
    :type payload: bytes
    :returns: The decoded image.
    :rtype: app.domain.SourceImage
    :raises app.domain.errors.ImageDecodeFailed: If Pillow cannot open
        the payload, including decompression-bomb refusals.
    """
    try:
        opened: Image = PilImage.open(BytesIO(payload)).convert(ImageMode.RGB)
    # Pillow raises DecompressionBombError, UnidentifiedImageError and
    # OSError among others; every one of them is the client's problem.
    except Exception as error:
        raise ImageDecodeFailed(detail=str(error)) from error
    return SourceImage(pixels=array(opened, dtype=uint8))


def encode_png(pixels: NDArray[uint8]) -> str:
    """PNG-encode an array and return it base64-encoded.

    :param pixels: 2D or 3D uint8 array.
    :type pixels: numpy.typing.NDArray[numpy.uint8]
    :returns: Base64 text of a PNG payload.
    :rtype: str
    """
    buffer: BytesIO
    with BytesIO() as buffer:
        PilImage.fromarray(pixels).save(buffer, format=ImageFormat.PNG)
        encoded: bytes = b64encode(buffer.getvalue())
    return encoded.decode(TextEncoding.UTF8)


@final
class OpenCvMaskDilator:
    """Grows a mask with OpenCV's rectangular structuring element."""

    # A method rather than a function because it implements the
    # MaskDilator capability, which the bootstrap supplies as an
    # instance.
    # pylint: disable=no-self-use
    def dilate(self, mask: MaskArray, percentage: float) -> MaskArray:
        """Return ``mask`` grown by ``percentage`` of its dimensions.

        :param mask: Mask to grow.
        :type mask: app.domain.MaskArray
        :param percentage: Kernel radius as a share of width and height.
        :type percentage: float
        :returns: The dilated mask, same shape as the input.
        :rtype: app.domain.MaskArray
        """
        ratio: float = clamp_percentage(percentage) / PercentageBounds.whole
        height, width = mask.shape[:2]
        kernel: NDArray[uint8] = ones(
            (
                int(height * ratio) + Dilation.minimum_kernel_size,
                int(width * ratio) + Dilation.minimum_kernel_size,
            ),
            dtype=uint8,
        )
        dilated: NDArray[uint8] = cv2_dilate(
            mask, kernel, iterations=Dilation.iterations
        ).astype(uint8)
        return dilated


__all__: list[str] = ["OpenCvMaskDilator", "decode_image", "encode_png"]
