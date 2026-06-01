from comic_text_detector.mlx_backend import backend as _backend
from comic_text_detector.mlx_backend.backend import MlxTextDetComputeBackend
from comic_text_detector.mlx_backend.configuration_textdet import TextDetConfig

importlib = _backend.importlib

__all__ = ["MlxTextDetComputeBackend", "TextDetConfig"]
