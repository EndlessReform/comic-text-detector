"""Compare MLX YOLO decoded block predictions against a detector fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from comic_text_detector.mlx_backend import MlxTextDetComputeBackend


YOLO_KEY = "head.yolo_decoded"


def compare_yolo(
    expected: np.lib.npyio.NpzFile,
    actual: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> tuple[dict[str, object], bool]:
    expected_array = expected[YOLO_KEY].astype(np.float32, copy=False)
    actual_array = actual.astype(np.float32, copy=False)
    diff = np.abs(actual_array - expected_array)
    allclose = bool(np.allclose(actual_array, expected_array, rtol=rtol, atol=atol))
    row = {
        "key": YOLO_KEY,
        "shape": list(actual_array.shape),
        "allclose": allclose,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "atol": atol,
        "rtol": rtol,
    }
    return row, allclose


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
        help="Fixture .npz containing input.nchw.fp32 and head.yolo_decoded.",
    )
    parser.add_argument("--atol", default=1e-3, type=float)
    parser.add_argument("--rtol", default=1e-3, type=float)
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
    actual = backend.forward_yolo_decoded(fixture["input.nchw.fp32"])
    row, passed = compare_yolo(fixture, actual, rtol=args.rtol, atol=args.atol)

    if args.json:
        print(json.dumps({"passed": passed, "yolo": row}, indent=2, sort_keys=True))
    else:
        status = "PASS" if row["allclose"] else "FAIL"
        print(
            f"{status}\t{row['key']}\tshape={tuple(row['shape'])}\t"
            f"max_abs={row['max_abs']:.8g}\tmean_abs={row['mean_abs']:.8g}"
        )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
