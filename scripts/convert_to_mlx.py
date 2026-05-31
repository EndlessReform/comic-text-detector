"""Inspect and convert comic-text-detector checkpoints for MLX work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_CHECKPOINT_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt"
)
SCHEMA = "comic_text_detector.mlx_conversion.v1"
DEFAULT_OUT_DIR = Path("output/detector/mlx-comictextdetector")
MODEL_FP32_FILENAME = "model.fp32.safetensors"
CONFIG_FILENAME = "config.json"


@dataclass(frozen=True)
class TensorInfo:
    key: str
    dtype: str
    shape: tuple[int, ...]
    elements: int
    bytes: int


@dataclass(frozen=True)
class ConvertedTensorInfo:
    key: str
    source_dtype: str
    output_dtype: str
    source_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    layout: str
    elements: int
    bytes: int


def default_checkpoint_path() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "manga-ocr"
    return cache_root / "comictextdetector.pt"


def download_url(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise RuntimeError(f"Failed downloading {url}")
        with out_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return out_path


def download_hf(repo_id: str, filename: str, revision: str | None = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading from Hugging Face requires huggingface-hub. "
            "Install comic-text-detector[mlx] or pass --checkpoint."
        ) from exc

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint

    if args.hf_repo_id is not None:
        return download_hf(args.hf_repo_id, args.hf_filename, revision=args.hf_revision)

    if args.url != DEFAULT_CHECKPOINT_URL:
        out_path = Path(tempfile.gettempdir()) / "comictextdetector.pt"
        return download_url(args.url, out_path)

    checkpoint = default_checkpoint_path()
    if checkpoint.is_file():
        return checkpoint
    return download_url(args.url, checkpoint)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def collect_tensors(value: Any, prefix: str = "") -> list[TensorInfo]:
    tensors = []
    if isinstance(value, torch.Tensor):
        tensors.append(
            TensorInfo(
                key=prefix,
                dtype=str(value.dtype).removeprefix("torch."),
                shape=tuple(value.shape),
                elements=value.numel(),
                bytes=tensor_nbytes(value),
            )
        )
    elif isinstance(value, torch.nn.Module):
        tensors.extend(collect_tensors(value.state_dict(), prefix=prefix))
    elif isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            tensors.extend(collect_tensors(value[key], prefix=child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            tensors.extend(collect_tensors(item, prefix=child_prefix))
    return tensors


def iter_tensors(value: Any, prefix: str = ""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, torch.nn.Module):
        yield from iter_tensors(value.state_dict(), prefix=prefix)
    elif isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_tensors(value[key], prefix=child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from iter_tensors(item, prefix=child_prefix)


def is_conv_transpose_weight(key: str) -> bool:
    return (
        key.startswith(("text_seg.upconv", "text_det.upconv")) and key.endswith(".conv.1.weight")
    ) or key in {
        "text_seg.upconv6.0.weight",
        "text_det.binarize.3.weight",
        "text_det.binarize.6.weight",
        "text_det.thresh.3.weight",
        "text_det.thresh.6.weight",
    }


def infer_layout_transform(key: str, tensor: torch.Tensor) -> str:
    if not key.endswith(".weight") or tensor.ndim != 4:
        return "none"
    if is_conv_transpose_weight(key):
        return "conv_transpose2d_iohw_to_ohwi"
    return "conv2d_oihw_to_ohwi"


def convert_tensor_for_mlx(key: str, tensor: torch.Tensor) -> tuple[torch.Tensor, str]:
    layout = infer_layout_transform(key, tensor)
    converted = tensor.detach().cpu()
    if layout == "conv2d_oihw_to_ohwi":
        converted = converted.permute(0, 2, 3, 1).contiguous()
    elif layout == "conv_transpose2d_iohw_to_ohwi":
        converted = converted.permute(1, 2, 3, 0).contiguous()

    if converted.is_floating_point():
        converted = converted.float()
    return converted.contiguous(), layout


def build_safetensors_payload(checkpoint: Any) -> tuple[dict[str, torch.Tensor], list[ConvertedTensorInfo]]:
    payload = {}
    infos = []
    for key, tensor in iter_tensors(checkpoint):
        converted, layout = convert_tensor_for_mlx(key, tensor)
        payload[key] = converted
        infos.append(
            ConvertedTensorInfo(
                key=key,
                source_dtype=str(tensor.dtype).removeprefix("torch."),
                output_dtype=str(converted.dtype).removeprefix("torch."),
                source_shape=tuple(tensor.shape),
                output_shape=tuple(converted.shape),
                layout=layout,
                elements=converted.numel(),
                bytes=tensor_nbytes(converted),
            )
        )
    return payload, infos


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_config(
    checkpoint: Any,
    checkpoint_path: Path,
    output_path: Path,
    infos: list[ConvertedTensorInfo],
) -> dict[str, Any]:
    transform_counts: dict[str, int] = {}
    for info in infos:
        transform_counts[info.layout] = transform_counts.get(info.layout, 0) + 1

    total_bytes = sum(info.bytes for info in infos)
    config = {
        "schema": SCHEMA,
        "source": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_url": DEFAULT_CHECKPOINT_URL,
            "format": "comictextdetector.pt",
        },
        "artifact": {
            "file": output_path.name,
            "dtype": "fp32",
            "tensor_count": len(infos),
            "total_elements": sum(info.elements for info in infos),
            "total_bytes": total_bytes,
            "total_mib": round(total_bytes / (1024 * 1024), 3),
        },
        "layouts": {
            "public_input": "NCHW",
            "mlx_internal_input": "NHWC",
            "conv2d_weight": "OHWI",
            "conv_transpose2d_weight": "OHWI",
            "conv2d_transform": "torch OIHW -> MLX OHWI",
            "conv_transpose2d_transform": "torch IOHW -> MLX OHWI",
        },
        "transform_counts": transform_counts,
        "tensors": [
            {
                "key": info.key,
                "source_dtype": info.source_dtype,
                "output_dtype": info.output_dtype,
                "source_shape": list(info.source_shape),
                "output_shape": list(info.output_shape),
                "layout": info.layout,
            }
            for info in infos
        ],
    }
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("blk_det"), dict):
        config["yolov5"] = {"cfg": jsonable(checkpoint["blk_det"].get("cfg"))}
    return config


def render_keys_report(checkpoint_path: Path, tensors: list[TensorInfo]) -> str:
    total_tensors = len(tensors)
    total_elements = sum(item.elements for item in tensors)
    total_bytes = sum(item.bytes for item in tensors)
    total_mib = total_bytes / (1024 * 1024)

    lines = [
        f"# checkpoint: {checkpoint_path}",
        "# key\tdtype\tshape\telements\tbytes",
    ]
    for item in tensors:
        shape = "(" + ", ".join(str(dim) for dim in item.shape) + ")"
        lines.append(f"{item.key}\t{item.dtype}\t{shape}\t{item.elements}\t{item.bytes}")

    lines.extend(
        [
            "",
            "# total model size",
            f"total_tensors\t{total_tensors}",
            f"total_elements\t{total_elements}",
            f"total_bytes\t{total_bytes}",
            f"total_mib\t{total_mib:.3f}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_keys(args: argparse.Namespace) -> None:
    checkpoint_path = resolve_checkpoint(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tensors = collect_tensors(checkpoint)
    report = render_keys_report(checkpoint_path, tensors)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote key report: {args.out}")
        print("\n".join(report.rstrip().splitlines()[-5:]))
    else:
        print(report, end="")


def run_dump(args: argparse.Namespace) -> None:
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dumping safetensors requires safetensors. Install comic-text-detector[mlx]."
        ) from exc

    checkpoint_path = resolve_checkpoint(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    payload, infos = build_safetensors_payload(checkpoint)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / MODEL_FP32_FILENAME
    config_path = args.out_dir / CONFIG_FILENAME
    save_file(
        payload,
        output_path,
        metadata={
            "schema": SCHEMA,
            "dtype": "fp32",
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
        },
    )
    config = build_config(checkpoint, checkpoint_path, output_path, infos)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote safetensors: {output_path}")
    print(f"wrote config: {config_path}")
    print(f"tensor_count\t{config['artifact']['tensor_count']}")
    print(f"total_mib\t{config['artifact']['total_mib']:.3f}")
    for layout, count in sorted(config["transform_counts"].items()):
        print(f"{layout}\t{count}")


def add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local comictextdetector.pt path. Defaults to mokuro's cached detector checkpoint.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_CHECKPOINT_URL,
        help="Checkpoint URL used when --checkpoint and --hf-repo-id are omitted.",
    )
    parser.add_argument("--hf-repo-id", help="Optional explicit Hugging Face detector repo id.")
    parser.add_argument("--hf-filename", default="comictextdetector.pt", help="Filename inside --hf-repo-id.")
    parser.add_argument("--hf-revision", help="Optional Hugging Face revision.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and convert comic-text-detector checkpoints for MLX.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    keys_parser = subparsers.add_parser("keys", help="Print checkpoint tensor keys, dims, and total size.")
    add_source_args(keys_parser)
    keys_parser.add_argument("--out", type=Path, help="Optional text report path.")
    keys_parser.set_defaults(func=run_keys)

    dump_parser = subparsers.add_parser("dump", help="Future safetensors conversion entrypoint.")
    add_source_args(dump_parser)
    dump_parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    dump_parser.add_argument("--dtype", default="fp32", choices=("fp32",), help="Output dtype; only fp32 is available.")
    dump_parser.set_defaults(func=run_dump)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
