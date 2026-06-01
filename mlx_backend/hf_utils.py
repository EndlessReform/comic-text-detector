"""Hugging Face Hub utilities for MLX detector artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

HF_HUB_PREFIX = "hf://"
DEFAULT_MLX_MODEL = "jkeisling/comictextdetector-mlx"


def _import_hf_hub() -> Any:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download
    except ModuleNotFoundError as exc:
        from comic_text_detector.backends import TextDetBackendUnavailable

        if exc.name and exc.name.split(".")[0] == "huggingface_hub":
            raise TextDetBackendUnavailable(
                "The MLX detector backend requires huggingface-hub for model downloads. "
                "Install comic-text-detector[mlx] or mokuro[mlx] to enable it."
            ) from exc
        raise


def is_hf_hub_path(path: str | Path) -> bool:
    """Check if a path string refers to a Hugging Face Hub repo."""
    s = str(path)
    return s.startswith(HF_HUB_PREFIX) or (
        "/" in s
        and not os.path.isabs(s)
        and not Path(s).is_file()
        and not Path(s).is_dir()
        and not s.startswith(".")
        and not s.startswith("/")
    )


def strip_hf_prefix(path: str | Path) -> str:
    """Strip the hf:// prefix if present, returning the repo id."""
    s = str(path)
    if s.startswith(HF_HUB_PREFIX):
        return s[len(HF_HUB_PREFIX) :]
    return s


def resolve_hf_model_path(
    model_path: str | Path | None,
    cache_dir: Path | None = None,
) -> Path:
    """Resolve a model path that may be a local path or HF Hub reference.

    Supports:
    - Local directories containing config.json + safetensors
    - Local .safetensors files
    - ``hf://username/repo`` syntax
    - Plain ``username/repo`` (auto-detected as HF repo if not a local path)
    - ``None`` (returns DEFAULT_MLX_MODEL which will be resolved on next call)

    Returns a Path pointing to the local artifact directory.
    """
    if model_path is None:
        return resolve_hf_model_path(DEFAULT_MLX_MODEL, cache_dir=cache_dir)

    repo_id = strip_hf_prefix(model_path)

    # If it looks like a local path that exists, return it directly
    local_path = Path(model_path).expanduser()
    if local_path.is_file() or local_path.is_dir():
        return local_path

    # Otherwise treat as HF Hub repo id and download
    snapshot_download = _import_hf_hub()
    local_dir = snapshot_download(repo_id, cache_dir=str(cache_dir) if cache_dir else None)
    return Path(local_dir)
