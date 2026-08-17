"""Pipeline behavior that needs the real models.

Slow: these load GroundingDINO and SAM 2.1 and run actual inference.
They are the checks that cannot be faked - that concurrent callers get
their own masks, and that the small models still segment correctly.

The fake-capability tests live in ``test_use_case.py``: those need no
model at all, which is the point.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from PIL import Image as PilImage

from app.bootstrap import Application, SegmentOutcome, build
from app.domain import BBox, Prompt, SegmentedImage, SourceImage, binarize
from app.settings import AuthMode, Settings

EXAMPLES: Final[Path] = Path(__file__).resolve().parents[1] / "examples"
#: Reference masks come from the previous SAM ViT-H pipeline, so this
#: doubles as the migration's regression gate.
MINIMUM_IOU: Final[float] = 0.85

pytestmark = pytest.mark.slow


@pytest.fixture(name="application", scope="module")
def application_fixture() -> Application:
    """Build the wired application once for the whole module.

    :returns: The assembled application, models loaded.
    :rtype: app.bootstrap.Application
    """
    settings: Settings = Settings(
        AUTH_MODE=AuthMode.NONE,
        MASK_PADDING_PCT=10.0,
        DILATION_PCT=3.0,
        _env_file=None,
    )
    return build(settings)


def _load(name: str) -> SourceImage:
    """Open one example image.

    :param name: Example stem, e.g. ``"dog"``.
    :type name: str
    :returns: The decoded image.
    :rtype: app.domain.models.SourceImage
    """
    opened = PilImage.open(EXAMPLES / f"{name}-original.png").convert("RGB")
    return SourceImage(pixels=np.array(opened, dtype=np.uint8))


def _segment(
    application: Application, name: str, prompt: str
) -> SegmentedImage:
    """Segment one example and require success.

    :param application: The wired application.
    :type application: app.bootstrap.Application
    :param name: Example stem.
    :type name: str
    :param prompt: Prompt to segment with.
    :type prompt: str
    :returns: The segmentation result.
    :rtype: app.domain.models.SegmentedImage
    """
    outcome: SegmentOutcome = application.segment(
        application.settings.default_segmenter,
        _load(name),
        Prompt.parse(prompt),
        None,
    )
    assert isinstance(outcome, SegmentedImage), f"{name}: {outcome!r}"
    return outcome


@pytest.mark.parametrize(
    ("name", "prompt"),
    [("dog", "dog"), ("cat", "cat"), ("man", "costume. glasses")],
)
def test_masks_match_the_reference_pipeline(
    application: Application, name: str, prompt: str
) -> None:
    """SAM 2.1-tiny reproduces what SAM ViT-H used to produce."""
    produced = binarize(_segment(application, name, prompt).cropped_mask)
    reference_image = PilImage.open(EXAMPLES / f"{name}-mask.png").convert("L")
    # Both masks are cropped to their own padded bbox; putting the
    # reference on the new crop's grid is what makes them comparable.
    reference = binarize(
        np.array(
            reference_image.resize(
                (produced.shape[1], produced.shape[0]), PilImage.NEAREST
            )
        )
    )
    union = np.logical_or(reference, produced).sum()
    iou = float(np.logical_and(reference, produced).sum()) / max(union, 1)
    assert iou >= MINIMUM_IOU, f"{name}: IoU {iou:.3f} < {MINIMUM_IOU}"


def test_scores_are_populated(application: Application) -> None:
    """Every detection carries both scores and a combined confidence."""
    result: SegmentedImage = _segment(application, "dog", "dog")
    assert result.detections, "expected at least one detection"
    for detection in result.detections:
        assert 0.0 <= detection.detection_score <= 1.0
        assert 0.0 <= detection.mask_score <= 1.0
        assert detection.confidence == pytest.approx(
            detection.detection_score * detection.mask_score
        )
    assert result.confidence == min(
        detection.confidence for detection in result.detections
    )


def test_concurrent_requests_do_not_cross_over(
    application: Application,
) -> None:
    """Parallel callers each get the mask for their own image.

    The pre-migration pipeline shared one predictor and mutated it with
    ``set_image()`` before reading it back, so two interleaving threads
    returned one caller's mask to the other, with no error raised.
    """
    cases: list[tuple[str, str]] = [("dog", "dog"), ("cat", "cat")]
    expected: dict[str, BBox] = {
        name: _segment(application, name, prompt).bbox
        for name, prompt in cases
    }

    def run_one(index: int) -> tuple[str, BBox]:
        """Segment one alternating example.

        :param index: Iteration counter deciding which case runs.
        :type index: int
        :returns: The case name and the box it produced.
        :rtype: tuple[str, app.domain.models.BBox]
        """
        name, prompt = cases[index % len(cases)]
        return name, _segment(application, name, prompt).bbox

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_one, range(8)))

    for name, bbox in results:
        assert bbox == expected[name], (
            f"{name} came back with another image's mask: "
            f"{bbox} != {expected[name]}"
        )
