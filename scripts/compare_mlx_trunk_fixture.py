"""Compare MLX YOLO trunk feature maps against a detector fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from comic_text_detector.mlx_backend import MlxTextDetComputeBackend


TRUNK_KEYS = tuple(f"trunk.feature_{index}" for index in (1, 3, 5, 7, 9))


def compare_features(
    expected: np.lib.npyio.NpzFile,
    actual: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> tuple[list[dict[str, object]], bool]:
    rows = []
    passed = True
    for key in TRUNK_KEYS:
        expected_feature = expected[key].astype(np.float32, copy=False)
        actual_feature = actual[key].astype(np.float32, copy=False)
        diff = np.abs(actual_feature - expected_feature)
        allclose = bool(np.allclose(actual_feature, expected_feature, rtol=rtol, atol=atol))
        passed = passed and allclose
        rows.append(
            {
                "key": key,
                "shape": list(actual_feature.shape),
                "allclose": allclose,
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "atol": atol,
                "rtol": rtol,
            }
        )
    return rows, passed


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
        help="Fixture .npz containing input.nchw.fp32 and trunk.feature_* arrays.",
    )
    parser.add_argument("--atol", default=1e-4, type=float)
    parser.add_argument("--rtol", default=1e-4, type=float)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture = np.load(args.fixture)
    backend = MlxTextDetComputeBackend(args.artifact)
    actual = backend.forward_trunk_features(fixture["input.nchw.fp32"])
    rows, passed = compare_features(fixture, actual, rtol=args.rtol, atol=args.atol)

    if args.json:
        print(json.dumps({"passed": passed, "features": rows}, indent=2, sort_keys=True))
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
