"""Compare MLX detector head outputs (mask + DB lines) against a fixture.

The trunk source is selectable so the head port can be validated in isolation:

* ``--trunk-source fixture`` (default) injects the Torch-reference
  ``trunk.feature_{1,3,5,7,9}`` arrays straight from the fixture ``.npz`` into the
  MLX heads. This isolates head numerics from any trunk drift -- if head outputs
  are wrong here, the bug is in the head port, not the trunk.
* ``--trunk-source mlx`` runs the full MLX pipeline (MLX trunk -> MLX heads) from
  ``input.nchw.fp32``, exercising the end-to-end stack.

The MLX CPU stream is the precision-preserving path (the heads contain 3x3
stride-1 C3 convs that hit the GPU Winograd fast path at high resolution and lose
fp32 accuracy, exactly like the trunk). It is slow at 1024x1024 but exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from comic_text_detector.mlx_backend import MlxTextDetComputeBackend


HEAD_KEYS = ("head.mask", "head.lines")
TRUNK_KEYS = tuple(f"trunk.feature_{index}" for index in (1, 3, 5, 7, 9))


def compare_heads(
    expected: np.lib.npyio.NpzFile,
    actual: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> tuple[list[dict[str, object]], bool]:
    rows = []
    passed = True
    for key in HEAD_KEYS:
        expected_array = expected[key].astype(np.float32, copy=False)
        actual_array = actual[key].astype(np.float32, copy=False)
        diff = np.abs(actual_array - expected_array)
        allclose = bool(np.allclose(actual_array, expected_array, rtol=rtol, atol=atol))
        passed = passed and allclose
        rows.append(
            {
                "key": key,
                "shape": list(actual_array.shape),
                "allclose": allclose,
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "atol": atol,
                "rtol": rtol,
            }
        )
    return rows, passed


def run_heads(
    backend: MlxTextDetComputeBackend,
    fixture: np.lib.npyio.NpzFile,
    trunk_source: str,
) -> dict[str, np.ndarray]:
    if trunk_source == "fixture":
        trunk_features = {key: fixture[key] for key in TRUNK_KEYS}
        return backend.forward_detector_heads(trunk_features)
    _yolo_decoded, mask, lines = backend.forward(fixture["input.nchw.fp32"])
    return {"head.mask": mask, "head.lines": lines}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        default=Path("output/detector/mlx-comictextdetector"),
        type=Path,
        help="Converted MLX artifact directory or model.fp32.safetensors path.",
    )
    parser.add_argument(
        "--fixture",
        default=Path("output/detector/single.npz"),
        type=Path,
        help="Fixture .npz with input.nchw.fp32, trunk.feature_*, head.mask, head.lines.",
    )
    parser.add_argument(
        "--trunk-source",
        default="fixture",
        choices=("fixture", "mlx"),
        help="'fixture' injects Torch-reference trunk features (head isolation); "
        "'mlx' runs the full MLX trunk+heads pipeline.",
    )
    parser.add_argument("--atol", default=1e-4, type=float)
    parser.add_argument("--rtol", default=1e-4, type=float)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "gpu", "default"),
        help="MLX compute device. 'cpu' is the precision-preserving path for parity.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture = np.load(args.fixture)
    compute_device = None if args.device == "default" else args.device
    backend = MlxTextDetComputeBackend(args.artifact, compute_device=compute_device)
    actual = run_heads(backend, fixture, args.trunk_source)
    rows, passed = compare_heads(fixture, actual, rtol=args.rtol, atol=args.atol)

    if args.json:
        print(json.dumps({"passed": passed, "trunk_source": args.trunk_source, "heads": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            status = "PASS" if row["allclose"] else "FAIL"
            print(
                f"{status}\t{row['key']}\tshape={tuple(row['shape'])}\t"
                f"max_abs={row['max_abs']:.8g}\tmean_abs={row['mean_abs']:.8g}"
            )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
