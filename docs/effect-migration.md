# Effect-style migration

Living record of the `SKILL.md` migration. Phase 0 inventory first, then the
mandated §53 error table and §54 capability inventory, kept current as modules
move.

## Phase 0 — inventory

Grepped per `SKILL.md` §37. What the repository does **not** have matters as
much as what it does:

| Pattern | Hits | Reading |
|---|---|---|
| `asyncio.create_task` / `gather` | 0 | No detached tasks to own |
| inline retry loops, `while True` | 0 | No ad-hoc retry to convert |
| HTTP/database clients | 0 | No connection lifetime to scope |
| `dict[str, Any]` | 0 | `JSONValue` is a closed recursive union |
| `except Exception` | 2 | `api/main.py:90`, `api/routes.py:107` |
| `raise ValueError` / `RuntimeError` | 3 | 2 defects, 1 boundary validation |
| ambient singletons | 6 | The real debt |

The service is synchronous and GPU-bound, so §68 applies: numeric work stays
plain Python and AnyIO buys nothing here. The migration is about **dependency
visibility and error precision**, not concurrency.

### Hidden dependencies to remove

| Global | Location | Becomes |
|---|---|---|
| `ModelManager` (`SingletonMeta`) | `managers/model_manager.py` | bootstrap-owned adapters |
| `SegmentationService` (`SingletonMeta`) | `services/segmentation_service.py` | `application/use_cases/segment.py` effect |
| `PersonService` (`SingletonMeta`) | `services/person_service.py` | `infrastructure/facelib/` adapter |
| `SEGMENTER_FACTORY` | `segmenters/factory.py` | typed mapping built in `bootstrap.py` |
| `INFERENCE_LOCK` | `managers/model_manager.py` | owned by the adapter that serialises the device |
| `get_settings()` (`lru_cache`) | `config/settings.py` | loaded once in `bootstrap.py`, narrow policies passed down |

`get_device()` / `get_model_dtype()` stay `lru_cache`d: they are pure hardware
interrogation, not operational dependencies (§14 — a global immutable value is
not a hidden dependency).

## §53 — error policy

| Error | Origin | Recoverable | Retryable | Public |
|---|---|---:|---:|---:|
| `InvalidPrompt` | boundary validation | yes | no | yes |
| `NoDetection` | detector adapter | yes | no | yes |
| `UnknownBackend` | boundary validation | yes | no | yes |
| `ImageDecodeFailed` | boundary validation | yes | no | yes |
| `UploadTooLarge` | boundary validation | yes | no | yes |
| `RateLimited` | interface policy | yes | yes, after delay | yes |
| `DeviceExhausted` | detector/refiner adapter | yes | yes, smaller input | yes |
| `ModelUnavailable` | adapter construction | yes | often | no |
| `FaceAnalysisUnavailable` | facelib adapter | yes | no | no |
| processor-not-loaded | invariant breach | **no — defect** | no | no |
| duplicate registration | invariant breach | **no — defect** | no | no |

Defects keep raising normally and surface as 500 with telemetry (§10, §13).

## §54 — capability inventory

| Capability | Used by | Implementation |
|---|---|---|
| `ObjectDetector` | `segment` | GroundingDINO via `transformers` |
| `MaskRefiner` | `segment` | SAM 2.1 Hiera-Tiny via `transformers` |
| `FaceAnalyser` | `analyse_person` | `facelib` fork (optional extra) |
| `SegmentationPolicy` | `segment` | frozen value object from settings |

Four capabilities, well inside `supply()`'s nine-argument ceiling.

## Migration report (`SKILL.md` §70)

| | |
|---|---|
| Files changed | 53, whole tree restructured to §42 |
| Capabilities introduced | `ObjectDetector`, `MaskRefiner`, `FaceAnalyser`, `MaskDilator` |
| Typed errors introduced | 9, listed above; no shared base to catch |
| Globals removed | 6 (`ModelManager`, `SegmentationService`, `PersonService`, `SEGMENTER_FACTORY`, `INFERENCE_LOCK`, `get_settings`) |
| Broad catches removed | 2 of 3; the survivor is the transport boundary in `interfaces/http/errors.py`, plus two narrow normalising catches in adapters |
| Ambiguous `None` resolved | `SegmentOutcome` replaces "returns `None` on failure"; `None` now means absence only |
| Resources scoped | Models owned by `bootstrap.build`, device work by `exclusive_device` |
| Detached tasks removed | 0 — there were none |
| Retry/timeout policies | None added: no network call survives in the request path |
| Validation boundaries | `Prompt.parse`, `decode_image`, `_read_upload`, `Settings` |
| Type-check status | mypy strict 0, pyright strict 0, pylint 10.00 |
| Test status | 24 passing, including 5 that need no model at all |

### Remaining legacy boundaries

None inside `src/app`. The two that remain are external: `facelib`, whose
absence is normalised at its adapter, and `transformers`, which is only
partially typed.

### Checker concessions, and why they are where they are

Every one is confined to a module whose job is to face an untyped or
awkward dependency:

| Concession | Location | Cause |
|---|---|---|
| 5 × `type-abstract` | `application/effects.py` | `need()` wants a concrete `type[T]`; a capability is a `Protocol` |
| 1 × `assignment` | `application/effects.py` | `catch` widens its result union to bare `Exception` |
| 1 × `reportArgumentType` | `application/wiring.py` | pyright cannot see `supply()` eliminating abilities structurally |
| 2 × `no-untyped-call` | adapters | `post_process_masks`, `eval()` carry no annotations |
| 3 × `reportCallIssue` | adapters | processor arguments declared through `**kwargs` upstream |
| `reportUnknown*` off | `infrastructure/` only | `transformers` is partially typed; scoped in `pyproject.toml` |

`pylint` and `pyright` disagree irreconcilably on Protocol method bodies:
pyright requires the `...` (a docstring-only body "falls off the end"),
pylint then calls the documented return redundant. pyright's complaint is
the substantive one, so the ellipsis stays and pylint yields, once, at the
top of `capabilities.py`.

### Known semantic ambiguities

Two calls taken from repository evidence rather than left open:

- the generic `Factory` is gone, but `GET /segmenters` and the
  `segmenter` field keep their contract through a typed mapping built in
  `bootstrap.py`;
- model loading is bootstrap-eager, which the readiness probe already
  encoded, so the observable startup contract did not change.

## Literal-extraction scope

The zero-magic-literal rule targets values that carry meaning. It deliberately
excludes:

- `__all__` entries — they mirror identifiers, and indirecting them would make
  the export list unreadable for no gain;
- docstrings and comments;
- the `constants.py` modules themselves, which are where literals live.

Everything else is in scope, including the ones easy to miss: `"RGB"`, `"PNG"`,
`"pt"`, mask values `255`/`127`, score bounds `0.0`/`1.0`, response dictionary
keys, and every Gradio label.
