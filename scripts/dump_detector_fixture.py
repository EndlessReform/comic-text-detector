"""Dump Torch detector fixtures for cross-backend parity tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from comic_text_detector.basemodel import TEXTDET_INFERENCE, TextDetBase
from comic_text_detector.inference import postprocess_mask, preprocess_img
from comic_text_detector.utils.db_utils import SegDetectorRepresenter
from comic_text_detector.utils.io_utils import NumpyEncoder, imread
from comic_text_detector.utils.textblock import group_output
from comic_text_detector.utils.textmask import REFINEMASK_INPAINT, refine_mask, refine_undetected_mask
from comic_text_detector.utils.yolov5_utils import non_max_suppression

SCHEMA = "comic_text_detector.mlx_fixture.v1"
BOX_THRESH = 0.6
DEFAULT_CHECKPOINT_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump Torch TextDetBase intermediate arrays and public detector outputs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to comictextdetector.pt. Defaults to the same cached detector mokuro uses.",
    )
    parser.add_argument("--image", required=True, type=Path, help="Image used for the single fixture.")
    parser.add_argument(
        "--batch-image",
        action="append",
        default=[],
        type=Path,
        help="Additional image for the batch fixture. May be passed more than once.",
    )
    parser.add_argument("--input-size", default=1024, type=int, help="Square detector input size.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for fixture files.")
    parser.add_argument("--activation", default="leaky", choices=("leaky", "relu"), help="Detector activation.")
    parser.add_argument("--device", default="cpu", help="Torch device. CPU is recommended for deterministic fixtures.")
    parser.add_argument("--conf-thresh", default=0.4, type=float, help="YOLO confidence threshold.")
    parser.add_argument("--nms-thresh", default=0.35, type=float, help="YOLO NMS IoU threshold.")
    parser.add_argument("--mask-thresh", default=0.3, type=float, help="Segmentation mask threshold.")
    parser.add_argument("--keep-undetected-mask", action="store_true", help="Mirror detector keep_undetected_mask=True.")
    parser.add_argument("--torch-threads", default=1, type=int, help="Torch CPU thread count.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_checkpoint_path() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "manga-ocr"
    return cache_root / "comictextdetector.pt"


def download_checkpoint_if_needed(path: Path) -> None:
    if path.is_file():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DEFAULT_CHECKPOINT_URL) as response:
        if response.status != 200:
            raise RuntimeError(f"Failed downloading {DEFAULT_CHECKPOINT_URL}")
        with path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def package_version(distribution: str, module: Any | None = None) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        if module is not None:
            return getattr(module, "__version__", None)
    return None


def detector_git_commit() -> str | None:
    package_dir = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "-C", str(package_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_image(path: Path) -> np.ndarray:
    image = imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def tensor_input_from_preprocessed(img_hwc: np.ndarray) -> np.ndarray:
    # Match preprocess_img(..., to_tensor=True) exactly while keeping the ndarray DMZ visible.
    return np.array([np.ascontiguousarray(img_hwc.transpose((2, 0, 1))[::-1])], dtype=np.float32) / 255.0


def load_inputs(paths: list[Path], input_size: tuple[int, int]) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    images = []
    nchw = []
    dw_dh = []
    resize_ratio = []

    for path in paths:
        image = read_image(path)
        images.append(image)
        img_in, _ratio, dw, dh = preprocess_img(image, input_size=input_size, to_tensor=False)
        im_h, im_w = image.shape[:2]
        nchw_one = tensor_input_from_preprocessed(img_in)
        nchw.append(nchw_one[0])
        dw_dh.append((dw, dh))
        resize_ratio.append((im_w / (input_size[0] - dw), im_h / (input_size[1] - dh)))

    input_nchw = np.stack(nchw, axis=0).astype(np.float32)
    return images, {
        "input.nchw.fp32": input_nchw,
        "input.nhwc.fp32": np.transpose(input_nchw, (0, 2, 3, 1)).astype(np.float32),
        "preprocess.dw_dh": np.asarray(dw_dh, dtype=np.int32),
        "preprocess.resize_ratio": np.asarray(resize_ratio, dtype=np.float32),
    }


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def forward_intermediates(
    net: TextDetBase,
    input_nchw: np.ndarray,
    device: str,
) -> dict[str, np.ndarray | torch.Tensor]:
    input_tensor = torch.from_numpy(input_nchw).to(device)
    with torch.no_grad():
        yolo_out, trunk_features = net.blk_det(input_tensor, detect=True)
        yolo_decoded = yolo_out[0] if isinstance(yolo_out, (tuple, list)) else yolo_out
        mask, seg_features = net.text_seg(*trunk_features, forward_mode=TEXTDET_INFERENCE)
        lines = net.text_det(*seg_features, step_eval=False)

    arrays: dict[str, np.ndarray | torch.Tensor] = {
        "head.yolo_decoded": as_numpy(yolo_decoded).astype(np.float32),
        "head.mask": as_numpy(mask).astype(np.float32),
        "head.lines": as_numpy(lines).astype(np.float32),
        "_head.yolo_decoded.torch": yolo_decoded,
        "_head.mask.torch": mask,
        "_head.lines.torch": lines,
    }
    for index, feature in zip((1, 3, 5, 7, 9), trunk_features):
        arrays[f"trunk.feature_{index}"] = as_numpy(feature).astype(np.float32)
    return arrays


def scaled_nms_values(
    decoded: torch.Tensor,
    conf_thresh: float,
    nms_thresh: float,
    resize_ratios: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    detections = non_max_suppression(decoded, conf_thresh, nms_thresh)
    per_image = []
    counts = []

    for det, resize_ratio in zip(detections, resize_ratios):
        det_np = as_numpy(det).astype(np.float32)
        if det_np.size:
            det_np[:, [0, 2]] *= resize_ratio[0]
            det_np[:, [1, 3]] *= resize_ratio[1]
        per_image.append(det_np)
        counts.append(det_np.shape[0])

    values = np.concatenate(per_image, axis=0) if per_image else np.zeros((0, 6), dtype=np.float32)
    if values.size == 0:
        values = np.zeros((0, 6), dtype=np.float32)
    return per_image, values.astype(np.float32), np.asarray(counts, dtype=np.int32)


def detector_blocks_from_nms(det_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blines = det_np[:, 0:4].astype(np.int32)
    confs = np.round(det_np[:, 4], 3)
    cls = det_np[:, 5].astype(np.int32)
    return blines, cls, confs


def postprocess_lines(
    seg_rep: SegDetectorRepresenter,
    input_size: tuple[int, int],
    lines_map: torch.Tensor,
    resize_ratios: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    line_values = []
    score_values = []
    counts = []

    for batch_idx, resize_ratio in enumerate(resize_ratios):
        lines, scores = seg_rep(input_size, lines_map[batch_idx : batch_idx + 1])
        idx = np.where(scores[0] > BOX_THRESH)
        lines_one = lines[0][idx]
        scores_one = scores[0][idx].astype(np.float32)
        if lines_one.size == 0:
            lines_one = np.zeros((0, 4, 2), dtype=np.int32)
        else:
            lines_one = lines_one.astype(np.float64)
            lines_one[..., 0] *= resize_ratio[0]
            lines_one[..., 1] *= resize_ratio[1]
            lines_one = lines_one.astype(np.int32)
        line_values.append(lines_one)
        score_values.append(scores_one)
        counts.append(lines_one.shape[0])

    all_lines = np.concatenate(line_values, axis=0) if line_values else np.zeros((0, 4, 2), dtype=np.int32)
    all_scores = np.concatenate(score_values, axis=0) if score_values else np.zeros((0,), dtype=np.float32)
    return line_values, score_values, all_lines.astype(np.int32), all_scores.astype(np.float32), np.asarray(counts, dtype=np.int32)


def final_json_entry(mask: np.ndarray, refined_mask: np.ndarray, blocks: list[Any]) -> dict[str, Any]:
    return {
        "mask": {
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
            "sha256": sha256_array(mask),
        },
        "refined_mask": {
            "shape": list(refined_mask.shape),
            "dtype": str(refined_mask.dtype),
            "sha256": sha256_array(refined_mask),
        },
        "blocks": [block.to_dict() for block in blocks],
    }


def enable_deterministic_torch() -> None:
    if not hasattr(torch, "use_deterministic_algorithms"):
        return
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def dump_split(
    name: str,
    paths: list[Path],
    net: TextDetBase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    input_size = (args.input_size, args.input_size)
    images, arrays = load_inputs(paths, input_size)
    intermediates = forward_intermediates(net, arrays["input.nchw.fp32"], args.device)
    arrays.update({key: value for key, value in intermediates.items() if not key.startswith("_")})

    decoded_torch = intermediates["_head.yolo_decoded.torch"]
    mask_torch = intermediates["_head.mask.torch"]
    lines_torch = intermediates["_head.lines.torch"]

    per_image_nms, nms_values, nms_counts = scaled_nms_values(
        decoded_torch,
        args.conf_thresh,
        args.nms_thresh,
        arrays["preprocess.resize_ratio"],
    )
    arrays["post.nms.values"] = nms_values
    arrays["post.nms.counts"] = nms_counts

    seg_rep = SegDetectorRepresenter(thresh=args.mask_thresh)
    per_image_lines, _scores_per_image, line_values, line_scores, line_counts = postprocess_lines(
        seg_rep,
        input_size,
        lines_torch,
        arrays["preprocess.resize_ratio"],
    )
    arrays["post.lines.values"] = line_values
    arrays["post.lines.scores"] = line_scores
    arrays["post.lines.counts"] = line_counts

    final_entries = []
    mask_uint8_pages = []
    for batch_idx, image in enumerate(images):
        im_h, im_w = image.shape[:2]
        dw, dh = arrays["preprocess.dw_dh"][batch_idx]
        mask_input = postprocess_mask(mask_torch[batch_idx].clone())
        mask_uint8_pages.append(mask_input)
        mask_one = mask_input[: mask_input.shape[0] - dh, : mask_input.shape[1] - dw]
        mask_one = cv2.resize(mask_one, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

        blocks = group_output(
            detector_blocks_from_nms(per_image_nms[batch_idx]),
            per_image_lines[batch_idx],
            im_w,
            im_h,
            mask_one,
        )
        refined_mask = refine_mask(image, mask_one, blocks, refine_mode=REFINEMASK_INPAINT)
        if args.keep_undetected_mask:
            refined_mask = refine_undetected_mask(
                image,
                mask_one,
                refined_mask,
                blocks,
                refine_mode=REFINEMASK_INPAINT,
            )
        final_entries.append(final_json_entry(mask_one, refined_mask, blocks))

    arrays["post.mask_uint8"] = np.stack(mask_uint8_pages, axis=0).astype(np.uint8)
    np.savez_compressed(args.out / f"{name}.npz", **arrays)
    with (args.out / f"{name}-final.json").open("w", encoding="utf8") as handle:
        json.dump(final_entries, handle, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        handle.write("\n")

    return {
        "name": name,
        "paths": [str(path) for path in paths],
        "npz": f"{name}.npz",
        "final_json": f"{name}-final.json",
        "num_images": len(paths),
        "nms_counts": nms_counts.tolist(),
        "line_counts": line_counts.tolist(),
        "block_counts": [len(entry["blocks"]) for entry in final_entries],
    }


def write_manifest(args: argparse.Namespace, splits: list[dict[str, Any]]) -> None:
    images = [args.image, *args.batch_image]
    manifest = {
        "schema": SCHEMA,
        "detector_git_commit": detector_git_commit(),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
        },
        "images": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in images
        ],
        "input_size": [args.input_size, args.input_size],
        "activation": args.activation,
        "dtype": "float32",
        "device": args.device,
        "thresholds": {
            "conf": args.conf_thresh,
            "nms": args.nms_thresh,
            "mask": args.mask_thresh,
            "line_box": BOX_THRESH,
        },
        "package_versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv-python": package_version("opencv-python", cv2) or cv2.__version__,
            "mlx": package_version("mlx"),
        },
        "splits": splits,
    }
    with (args.out / "manifest.json").open("w", encoding="utf8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser() if args.checkpoint is not None else default_checkpoint_path()
    args.image = args.image.expanduser()
    args.batch_image = [path.expanduser() for path in args.batch_image]
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    download_checkpoint_if_needed(args.checkpoint)

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(0)
    enable_deterministic_torch()

    net = TextDetBase(str(args.checkpoint), device=args.device, act=args.activation)
    net.eval()

    batch_paths = [args.image, *(args.batch_image or [args.image])]
    splits = [
        dump_split("single", [args.image], net, args),
        dump_split("batch", batch_paths, net, args),
    ]
    write_manifest(args, splits)


if __name__ == "__main__":
    main()
