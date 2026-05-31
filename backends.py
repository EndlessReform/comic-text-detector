from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from comic_text_detector.basemodel import TextDetBase


class TextDetBackendUnavailable(RuntimeError):
    """Raised when a requested detector backend cannot be constructed."""


class TextDetComputeBackend(Protocol):
    name: str

    def forward(self, input_nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return yolo_decoded, mask, and lines as numpy arrays."""


def _as_float32_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


class TorchTextDetComputeBackend:
    name = "torch"

    def __init__(self, model_path: str | Path, device: str = "cpu", half: bool = False, act: str = "leaky"):
        self.device = device
        self.half = half
        self.net = TextDetBase(model_path, device=device, half=half, act=act)

    @torch.no_grad()
    def forward(self, input_nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = torch.from_numpy(np.ascontiguousarray(input_nchw)).to(self.device)
        if self.half:
            input_tensor = input_tensor.half()
        yolo_decoded, mask, lines = self.net(input_tensor)
        return _as_float32_array(yolo_decoded), _as_float32_array(mask), _as_float32_array(lines)


def create_compute_backend(
    backend: str,
    model_path: str | Path,
    device: str = "cpu",
    half: bool = False,
    act: str = "leaky",
) -> TextDetComputeBackend:
    if backend == "torch":
        return TorchTextDetComputeBackend(model_path=model_path, device=device, half=half, act=act)
    if backend == "mlx":
        from comic_text_detector.mlx_backend import MlxTextDetComputeBackend

        return MlxTextDetComputeBackend(model_path=model_path)
    raise ValueError(f"unknown text detector compute backend: {backend}")
