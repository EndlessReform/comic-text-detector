from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from comic_text_detector.backends import TextDetBackendUnavailable


class MlxTextDetComputeBackend:
    name = "mlx"

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.mx = self._import_mlx()
        raise TextDetBackendUnavailable(
            "The MLX detector backend is present as a stub, but model conversion and forward execution "
            "are not implemented yet."
        )

    @staticmethod
    def _import_mlx():
        try:
            return importlib.import_module("mlx.core")
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] == "mlx":
                raise TextDetBackendUnavailable(
                    "The MLX detector backend requires the optional mlx extra. "
                    "Install comic-text-detector[mlx] or mokuro[mlx] to enable it."
                ) from exc
            raise

    def forward(self, input_nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise TextDetBackendUnavailable(
            "The MLX detector backend is present as a stub, but model forward execution is not implemented yet."
        )
