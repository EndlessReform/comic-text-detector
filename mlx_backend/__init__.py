import importlib

from comic_text_detector.mlx_backend.backend import MlxTextDetComputeBackend
from comic_text_detector.mlx_backend.configuration_textdet import TextDetConfig
from comic_text_detector.mlx_backend.hf_utils import (
    DEFAULT_MLX_MODEL,
    is_hf_hub_path,
    resolve_hf_model_path,
    strip_hf_prefix,
)

__all__ = [
    "MlxTextDetComputeBackend",
    "TextDetConfig",
    "DEFAULT_MLX_MODEL",
    "importlib",
    "is_hf_hub_path",
    "resolve_hf_model_path",
    "strip_hf_prefix",
]
