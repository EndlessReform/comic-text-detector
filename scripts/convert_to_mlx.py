"""Inspect and convert comic-text-detector checkpoints for MLX work."""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class TensorInfo:
    key: str
    dtype: str
    shape: tuple[int, ...]
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
    raise NotImplementedError("safetensors dumping will be added after inspecting checkpoint keys and dims")


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
    dump_parser.add_argument("--out-dir", required=True, type=Path)
    dump_parser.set_defaults(func=run_dump)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
