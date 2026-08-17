"""Face-analysis adapter: YuNet detection plus two small classifiers.

Implements :class:`~app.application.capabilities.FaceAnalyser`.

Replaces the ``facelib`` fork, which dragged scikit-image, scikit-learn,
scipy and tqdm into the tree for one optional feature. What runs here
instead:

* **detection** - YuNet through ``cv2.FaceDetectorYN``. OpenCV is already
  a dependency, and the model is a 232 KB ONNX file under the MIT
  licence, so this costs no new runtime and almost no memory.
* **age and gender** - two ViT/SigLIP classifiers, ~86M and ~93M
  parameters, both Apache-2.0, both served through ``transformers`` in
  half precision like every other model here.

One deliberate behaviour change is recorded in
:func:`app.domain.rules.certainly_adult`: the age estimator classifies
into bands rather than years, and the adult threshold falls inside one
of them. That band never certifies adulthood, which is stricter than the
numeric comparison it replaced.
"""

from typing import Final, final

from cv2 import COLOR_RGB2BGR, FaceDetectorYN, cvtColor
from huggingface_hub import hf_hub_download
from numpy import uint8
from numpy.typing import NDArray
from transformers import (
    SiglipForImageClassification,
    SiglipImageProcessor,
    ViTForImageClassification,
    ViTImageProcessor,
)

from app.application.policies import FacePolicy
from app.domain import (
    AgeBand,
    certainly_adult,
    Gender,
    ModelUnavailable,
    PersonPayload,
    SourceImage,
)
from app.infrastructure.constants import (
    AGE_ESTIMATOR,
    FACE,
    FACE_DETECTOR,
    GENDER_ESTIMATOR,
    TensorFormat,
)
from app.infrastructure.device import get_device, get_model_dtype
from app.infrastructure.execution import exclusive_device, fetching, place

_OPERATION: Final[str] = "analyse-faces"
_FIRST_RESULT: Final[int] = 0

#: The published label of each estimator, mapped onto the domain's own
#: vocabulary so no model string reaches the application layer.
_AGE_LABELS: Final[dict[str, AgeBand]] = {
    "0-2": AgeBand.INFANT,
    "3-9": AgeBand.CHILD,
    "10-19": AgeBand.TEEN,
    "20-29": AgeBand.TWENTIES,
    "30-39": AgeBand.THIRTIES,
    "40-49": AgeBand.FORTIES,
    "50-59": AgeBand.FIFTIES,
    "60-69": AgeBand.SIXTIES,
    "more than 70": AgeBand.SEVENTIES_PLUS,
}
_MALE_LABEL: Final[str] = "male portrait"

type Preprocessor = ViTImageProcessor | SiglipImageProcessor
type Classifier = ViTForImageClassification | SiglipForImageClassification


def _winning_label(
    processor: Preprocessor, model: Classifier, crop: NDArray[uint8]
) -> str:
    """Return the label one classifier scores highest for a crop.

    :param processor: Matching preprocessor.
    :type processor: Preprocessor
    :param model: The classifier.
    :type model: Classifier
    :param crop: Face pixels.
    :type crop: numpy.typing.NDArray[numpy.uint8]
    :returns: The winning label.
    :rtype: str
    :raises app.domain.errors.ModelUnavailable: If the checkpoint
        carries no label mapping, which makes its output meaningless.
    """
    features = processor(images=crop, return_tensors=TensorFormat.PYTORCH)
    device, dtype = get_device(), get_model_dtype()
    outputs = model(**features.to(device, dtype))
    index: int = int(outputs.logits.float().argmax(-1)[_FIRST_RESULT])
    if (labels := model.config.id2label) is None:
        raise ModelUnavailable(
            model=type(model).__name__,
            detail="The checkpoint carries no label mapping.",
        )
    # Checkpoints key this mapping by the index or by its string
    # form; normalising once removes the ambiguity for good.
    by_index: dict[str, str] = {
        str(key): value for key, value in labels.items()
    }
    if (found := by_index.get(str(index))) is None:
        raise ModelUnavailable(
            model=type(model).__name__,
            detail=f"No label for class {index}.",
        )
    return found


@final
class YuNetFaceAnalyser:
    """Detects faces, then classifies each crop's age band and gender."""

    def __init__(
        self,
        *,
        detector: FaceDetectorYN,
        age_processor: ViTImageProcessor,
        age_model: ViTForImageClassification,
        gender_processor: SiglipImageProcessor,
        gender_model: SiglipForImageClassification,
        policy: FacePolicy,
    ) -> None:
        """Bind already-constructed components.

        :param detector: Configured YuNet detector.
        :type detector: cv2.FaceDetectorYN
        :param age_processor: Preprocessor of the age estimator.
        :type age_processor: transformers.ViTImageProcessor
        :param age_model: The age estimator.
        :type age_model: transformers.ViTForImageClassification
        :param gender_processor: Preprocessor of the gender estimator.
        :type gender_processor: transformers.SiglipImageProcessor
        :param gender_model: The gender estimator.
        :type gender_model: transformers.SiglipForImageClassification
        :param policy: Detection tuning.
        :type policy: app.application.policies.FacePolicy
        """
        self._detector: FaceDetectorYN = detector
        self._age_processor: ViTImageProcessor = age_processor
        self._age_model: ViTForImageClassification = age_model
        self._gender_processor: SiglipImageProcessor = gender_processor
        self._gender_model: SiglipForImageClassification = gender_model
        self._policy: FacePolicy = policy

    def _crops(self, image: SourceImage) -> list[NDArray[uint8]]:
        """Return one pixel crop per detected face.

        :param image: Source image.
        :type image: app.domain.models.SourceImage
        :returns: Crops large enough to classify, possibly empty.
        :rtype: list[numpy.typing.NDArray[numpy.uint8]]
        """
        bgr = cvtColor(image.pixels, COLOR_RGB2BGR)
        self._detector.setScoreThreshold(self._policy.score_threshold)
        self._detector.setInputSize((image.width, image.height))
        _, detections = self._detector.detect(bgr)
        # YuNet answers with ``None`` rather than an empty array when
        # it finds nothing - measured, not assumed. The OpenCV stubs
        # declare the return non-optional, hence the unreachable ignore:
        # dropping the guard would crash on every face-free image.
        if detections is None:  # pyright: ignore[reportUnnecessaryComparison]
            return []  # type: ignore[unreachable]
        crops: list[NDArray[uint8]] = []
        for row in detections:
            left, top, width, height = (int(value) for value in row[:4])
            if min(width, height) < FACE.minimum_crop_pixels:
                continue
            left, top = max(0, left), max(0, top)
            crops.append(image.pixels[top : top + height, left : left + width])
        return crops

    def analyse(self, image: SourceImage) -> PersonPayload:
        """Summarise every face found in the image.

        An image with no face yields an empty payload whose ``is_adult``
        is vacuously true, which is the contract the API has always
        exposed.

        :param image: Source image.
        :type image: app.domain.models.SourceImage
        :returns: Gender codes, age bands and adulthood across all faces.
        :rtype: app.domain.models.PersonPayload
        :raises app.domain.errors.DeviceExhausted: On accelerator
            out-of-memory.
        """
        # Detection is OpenCV on the host, but the two classifiers are
        # device work like any other, so the whole pass takes the lock.
        with exclusive_device(_OPERATION):
            genders: list[Gender] = []
            bands: list[AgeBand] = []
            for crop in self._crops(image):
                age_label: str = _winning_label(
                    self._age_processor, self._age_model, crop
                )
                gender_label: str = _winning_label(
                    self._gender_processor, self._gender_model, crop
                )
                bands.append(_AGE_LABELS.get(age_label, AgeBand.TEEN))
                genders.append(
                    Gender.MALE
                    if gender_label == _MALE_LABEL
                    else Gender.FEMALE
                )
        # ``certainly_adult`` is the domain's call, not the adapter's.
        return PersonPayload(
            genders=tuple(genders),
            age_bands=tuple(bands),
            is_adult=certainly_adult(tuple(bands)),
        )


def load_face_analyser(policy: FacePolicy) -> YuNetFaceAnalyser:
    """Build the face analyser, downloading its weights if needed.

    :param policy: Detection tuning.
    :type policy: app.application.policies.FacePolicy
    :returns: A ready analyser.
    :rtype: YuNetFaceAnalyser
    :raises app.domain.errors.ModelUnavailable: If any artefact cannot
        be fetched or built.
    """
    weights: str = fetching(
        FACE_DETECTOR.repository,
        lambda: hf_hub_download(
            FACE_DETECTOR.repository,
            FACE_DETECTOR.filename,
            revision=FACE_DETECTOR.revision,
        ),
    )
    try:
        detector: FaceDetectorYN = FaceDetectorYN.create(
            weights,
            "",
            FACE.initial_size,
            score_threshold=policy.score_threshold,
            nms_threshold=FACE.nms_threshold,
            top_k=FACE.top_k,
        )
    except Exception as error:
        raise ModelUnavailable(
            model=FACE_DETECTOR.repository, detail=str(error)
        ) from error

    age_processor = fetching(
        AGE_ESTIMATOR.repository,
        lambda: ViTImageProcessor.from_pretrained(
            AGE_ESTIMATOR.repository, revision=AGE_ESTIMATOR.revision
        ),
    )
    age_model = fetching(
        AGE_ESTIMATOR.repository,
        lambda: ViTForImageClassification.from_pretrained(
            AGE_ESTIMATOR.repository,
            revision=AGE_ESTIMATOR.revision,
            dtype=get_model_dtype(),
        ),
    )
    gender_processor = fetching(
        GENDER_ESTIMATOR.repository,
        lambda: SiglipImageProcessor.from_pretrained(
            GENDER_ESTIMATOR.repository,
            revision=GENDER_ESTIMATOR.revision,
        ),
    )
    gender_model = fetching(
        GENDER_ESTIMATOR.repository,
        lambda: SiglipForImageClassification.from_pretrained(
            GENDER_ESTIMATOR.repository,
            revision=GENDER_ESTIMATOR.revision,
            dtype=get_model_dtype(),
        ),
    )
    return YuNetFaceAnalyser(
        detector=detector,
        age_processor=age_processor,
        age_model=place(age_model),
        gender_processor=gender_processor,
        gender_model=place(gender_model),
        policy=policy,
    )


__all__: list[str] = ["YuNetFaceAnalyser", "load_face_analyser"]
