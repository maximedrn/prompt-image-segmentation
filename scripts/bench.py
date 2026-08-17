"""Measure VRAM, latency and startup cost of the segmentation pipeline.

Drives the wired application built by :func:`app.bootstrap.build`, so it
measures exactly what production runs. Capture a baseline, then re-run
after any change that could move the budget.

Usage::

    poetry run python scripts/bench.py --output bench/baseline.json
    poetry run python scripts/bench.py --output bench/after-phase-3.json

The interesting number is ``memory.nvidia_smi_bytes``: that is what the
<1 GB target is measured against. ``memory.peak_device_bytes`` excludes the
~280 MB CUDA context, so it always reads lower.
"""

# Imports are deliberately split: the clock starts before torch is pulled
# in, because "how long until the first mask" includes importing torch.
from time import monotonic

_PROCESS_START: float = monotonic()

# pylint: disable=wrong-import-position
import argparse
import json
import os
import resource
import subprocess
import sys
from math import ceil
from pathlib import Path
from typing import Any, Final

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

_SCHEMA_VERSION: Final[int] = 1
_TARGET_BYTES: Final[int] = 1024**3
_MIB: Final[int] = 1024**2

# Prompts mirror the README example table.
_SAMPLES: Final[tuple[tuple[str, str], ...]] = (
    ("dog-original.png", "dog"),
    ("cat-original.png", "cat"),
    ("man-original.png", "costume. glasses"),
)


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, correct for a single sample.

    :param values: Measured durations, any order.
    :type values: list[float]
    :param fraction: Target quantile in ``[0, 1]``.
    :type fraction: float
    :returns: The value at that rank, or ``0.0`` when empty.
    :rtype: float
    """
    if not values:
        return 0.0
    ordered: list[float] = sorted(values)
    rank: int = ceil(fraction * len(ordered)) - 1
    return ordered[min(max(rank, 0), len(ordered) - 1)]


def _host_peak_bytes() -> int:
    """Peak resident set size of this process.

    :returns: Peak RSS in bytes.
    :rtype: int
    """
    peak: int = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # getrusage reports bytes on macOS but kilobytes on Linux.
    return peak if sys.platform == "darwin" else peak * 1024


def _nvidia_smi_bytes() -> int | None:
    """VRAM attributed to this PID by the driver, contexts included.

    This is the figure the <1 GB target is stated against; the torch
    allocator cannot see the CUDA context, so it under-reports.

    :returns: Bytes used on the GPU, or ``None`` off NVIDIA hardware.
    :rtype: int | None
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except OSError, subprocess.SubprocessError:
        return None
    pid: str = str(os.getpid())
    for line in completed.stdout.splitlines():
        fields: list[str] = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == pid and fields[1].isdigit():
            return int(fields[1]) * _MIB
    return None


def main() -> int:
    """Run the benchmark and emit the report.

    :returns: ``0`` when every sample succeeded, ``1`` otherwise.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    arguments = parser.parse_args()

    import numpy  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from app.bootstrap import build  # noqa: PLC0415
    from app.domain import Prompt, SourceImage  # noqa: PLC0415
    from app.infrastructure.device import get_device  # noqa: PLC0415
    from app.settings import AuthMode, Settings  # noqa: PLC0415

    import_seconds: float = monotonic() - _PROCESS_START
    device = get_device()

    def peak_device_bytes() -> int | None:
        """Peak device memory the torch allocator knows about.

        :returns: Bytes, or ``None`` on CPU where the notion is host RSS.
        :rtype: int | None
        """
        if device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(device))
        if device.type == "mps":
            return int(torch.mps.driver_allocated_memory())
        return None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    load_start: float = monotonic()
    application = build(Settings(AUTH_MODE=AuthMode.NONE, _env_file=None))
    load_seconds: float = monotonic() - load_start
    after_load_bytes: int | None = peak_device_bytes()

    backend: str = application.settings.default_segmenter
    durations: list[float] = []
    failures: list[dict[str, str]] = []
    first_mask_seconds: float | None = None

    for _ in range(arguments.runs):
        for filename, default_prompt in _SAMPLES:
            path: Path = _PROJECT_ROOT / "examples" / filename
            if not path.is_file():
                failures.append({"sample": filename, "error": "missing"})
                continue
            prompt: str = arguments.prompt or default_prompt
            opened = Image.open(path).convert("RGB")
            image = SourceImage(pixels=numpy.array(opened, dtype=numpy.uint8))
            started: float = monotonic()
            try:
                application.segment(backend, image, Prompt.parse(prompt), None)
            # A benchmark must survive a bad sample and report it.
            except Exception as error:  # pylint: disable=broad-except
                failures.append(
                    {"sample": filename, "error": f"{type(error).__name__}"}
                )
                continue
            durations.append(monotonic() - started)
            if first_mask_seconds is None:
                first_mask_seconds = monotonic() - _PROCESS_START

    report: dict[str, Any] = {
        "schema": _SCHEMA_VERSION,
        "device": str(device),
        "torch": torch.__version__,
        "startup": {
            "import_seconds": round(import_seconds, 3),
            "load_seconds": round(load_seconds, 3),
            "first_mask_seconds": (
                round(first_mask_seconds, 3)
                if first_mask_seconds is not None
                else None
            ),
        },
        "memory": {
            "after_load_device_bytes": after_load_bytes,
            "peak_device_bytes": peak_device_bytes(),
            "nvidia_smi_bytes": _nvidia_smi_bytes(),
            "host_peak_bytes": _host_peak_bytes(),
        },
        "latency": {
            "runs": len(durations),
            "p50_seconds": round(_percentile(durations, 0.50), 3),
            "p95_seconds": round(_percentile(durations, 0.95), 3),
        },
        "failures": failures,
    }

    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))

    measured: int | None = report["memory"]["nvidia_smi_bytes"]
    if measured is not None:
        verdict: str = "PASS" if measured < _TARGET_BYTES else "OVER BUDGET"
        print(
            f"\nnvidia-smi: {measured / _MIB:.0f} MiB "
            f"(target < {_TARGET_BYTES / _MIB:.0f} MiB) -> {verdict}",
            file=sys.stderr,
        )
    else:
        print(
            "\nnvidia-smi unavailable: the <1 GB target is a CUDA figure, "
            "so this run only measures latency and host memory.",
            file=sys.stderr,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
