"""Encoding / decoding: PIL <-> torch tensor <-> base64 PNG."""

from base64 import b64encode
from io import BytesIO

from PIL import Image as PilImage
from PIL.Image import Image
from groundingdino.datasets.transforms import (
    Compose,
    Normalize,
    RandomResize,
    ToTensor,
)
from numpy import array, uint8
from numpy.typing import NDArray
from torch import Tensor

from app.config import get_device


def pil_to_dino_tensor(image: Image) -> Tensor:
    """Transform a PIL image into the tensor GroundingDINO expects.

    Applies the reference ``RandomResize -> ToTensor -> Normalize``
    pipeline, then moves the result to the current torch device.

    :param image: Source RGB image.
    :type image: PIL.Image.Image
    :returns: A 3xHxW float tensor on the process-wide torch device.
    :rtype: torch.Tensor
    """
    transform: Compose = Compose([
        RandomResize([800], max_size=1333),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor: Tensor = transform(image, None)[0]
    return tensor.to(get_device())


def tensor_to_uint8_array(tensor: Tensor) -> NDArray[uint8]:
    """Detach a CUDA/MPS tensor to a host uint8 ndarray.

    :param tensor: Any-shape torch tensor, already in ``uint8`` range.
    :type tensor: torch.Tensor
    :returns: A numpy view of the tensor after moving to host memory.
    :rtype: numpy.typing.NDArray[numpy.uint8]
    """
    return tensor.cpu().numpy()


def array_to_pil(source: NDArray[uint8]) -> Image:
    """Wrap an in-memory uint8 array as a PIL image.

    :param source: HxWxC (or HxW) uint8 ndarray.
    :type source: numpy.typing.NDArray[numpy.uint8]
    :returns: A PIL image sharing the underlying buffer.
    :rtype: PIL.Image.Image
    """
    return PilImage.fromarray(source)


def pil_to_array(image: Image) -> NDArray[uint8]:
    """Copy a PIL image into a uint8 ndarray.

    :param image: Source PIL image.
    :type image: PIL.Image.Image
    :returns: A freshly allocated ndarray in the image's native layout.
    :rtype: numpy.typing.NDArray[numpy.uint8]
    """
    return array(image, dtype=uint8)


def pil_to_base64(image: Image) -> str:
    """PNG-encode ``image`` and return the base64 ASCII string.

    :param image: Image to serialize.
    :type image: PIL.Image.Image
    :returns: Base64 (UTF-8) representation of a PNG payload.
    :rtype: str
    """
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        payload: bytes = buffer.getvalue()
    return b64encode(payload).decode("utf-8")


__all__: list[str] = [
    "pil_to_dino_tensor",
    "tensor_to_uint8_array",
    "array_to_pil",
    "pil_to_array",
    "pil_to_base64",
]
