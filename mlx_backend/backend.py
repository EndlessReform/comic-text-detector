from __future__ import annotations

import contextlib
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from comic_text_detector.backends import TextDetBackendUnavailable
from comic_text_detector.mlx_backend.configuration_textdet import CONFIG_FILENAME, TextDetConfig
from comic_text_detector.mlx_backend.hf_utils import resolve_hf_model_path
from comic_text_detector.mlx_backend.modeling_textdet import MlxComicTextDetector


COMPUTE_DEVICES = ("gpu", "cpu")
COMPUTE_DTYPES = ("float32", "bf16")


class MlxTextDetComputeBackend:
    name = "mlx"

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        compute_device: str | None = None,
        compute_dtype: str | None = None,
        compile_model: bool = False,
        hf_cache_dir: Path | None = None,
    ):
        # Import mlx first (before any HF hub resolution that might trigger importlib)
        self.mx = self._import_mlx()
        self._hf_cache_dir = hf_cache_dir
        resolved_path = resolve_hf_model_path(model_path, cache_dir=hf_cache_dir)
        self.model_path = resolved_path
        self.compute_device = compute_device
        self.compute_dtype_name, self.compute_dtype = self._resolve_dtype(self.mx, compute_dtype)
        self.compile_model = compile_model
        self.stream = self._resolve_stream(self.mx, compute_device)
        self.config_path = self._resolve_config_path(self.model_path)
        self.config = TextDetConfig.from_json_file(self.config_path)
        self.weights_path = self._resolve_weights_path(self.model_path, self.config)
        self.weights = self.mx.load(str(self.weights_path))
        self.model = MlxComicTextDetector(self.mx, self.config, self.weights)
        if self.compute_dtype_name != "float32":
            self.model.set_dtype(self.compute_dtype)
        self.compiled_block_count = self.model.compile_conv_blocks() if compile_model else 0

    @staticmethod
    def _resolve_stream(mx: Any, compute_device: str | None) -> Any | None:
        """Map a device name to an MLX stream."""
        if compute_device is None:
            return None
        if compute_device not in COMPUTE_DEVICES:
            raise TextDetBackendUnavailable(
                f"Unsupported MLX compute device {compute_device!r}; "
                f"expected one of {COMPUTE_DEVICES}."
            )
        device = mx.cpu if compute_device == "cpu" else mx.gpu
        return mx.new_stream(device)

    @staticmethod
    def _resolve_dtype(mx: Any, compute_dtype: str | None) -> tuple[str, Any]:
        if compute_dtype is None:
            return "float32", mx.float32
        aliases = {"fp32": "float32", "float32": "float32", "bf16": "bf16", "bfloat16": "bf16"}
        dtype_name = aliases.get(compute_dtype)
        if dtype_name not in COMPUTE_DTYPES:
            raise TextDetBackendUnavailable(
                f"Unsupported MLX compute dtype {compute_dtype!r}; "
                f"expected one of {COMPUTE_DTYPES}."
            )
        return dtype_name, mx.bfloat16 if dtype_name == "bf16" else mx.float32

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

    @staticmethod
    def _resolve_config_path(model_path: Path) -> Path:
        if model_path.is_dir():
            config_path = model_path / CONFIG_FILENAME
            if config_path.is_file():
                return config_path
        # Check parent dir for config (e.g. hf_hub nested layouts)
        if not model_path.is_dir():
            config_path = model_path.parent / CONFIG_FILENAME
            if config_path.is_file():
                return config_path
        # Walk up to find config in a parent directory (hf_hub caches can nest repos)
        current = model_path
        for _ in range(5):
            parent_config = current / CONFIG_FILENAME
            if parent_config.is_file():
                return parent_config
            parent = current.parent
            if parent == current:
                break
            current = parent

        raise TextDetBackendUnavailable(
            f"MLX detector config (config.json) not found near: {model_path}"
        )

    @staticmethod
    def _resolve_weights_path(model_path: Path, config: TextDetConfig) -> Path:
        # Try model_path directly (directory or file)
        if model_path.is_dir():
            weights_path = model_path / config.weights.file
            if weights_path.is_file():
                return weights_path
        elif model_path.suffix == ".safetensors":
            return model_path

        # Search in parent directories (for hf_hub nested cache layouts)
        current = model_path.parent if not model_path.is_dir() else model_path
        for _ in range(5):
            weights_path = current / config.weights.file
            if weights_path.is_file():
                return weights_path
            parent = current.parent
            if parent == current:
                break
            current = parent

        raise TextDetBackendUnavailable(
            f"MLX detector weights ({config.weights.file}) not found near: {model_path}"
        )

    def _stream_ctx(self):
        if self.stream is None:
            return contextlib.nullcontext()
        return self.mx.stream(self.stream)

    def _input_to_mx(self, input_nchw: np.ndarray) -> Any:
        if input_nchw.ndim != 4 or input_nchw.shape[1] != 3:
            raise ValueError(
                f"Expected NCHW input with 3 channels, got shape {input_nchw.shape}"
            )
        return self.mx.array(
            np.ascontiguousarray(input_nchw, dtype=np.float32)
        ).transpose(0, 2, 3, 1).astype(self.compute_dtype)

    def _feature_to_mx(self, feature_nchw: np.ndarray) -> Any:
        if feature_nchw.ndim != 4:
            raise ValueError(f"Expected NCHW feature tensor, got shape {feature_nchw.shape}")
        return self.mx.array(
            np.ascontiguousarray(feature_nchw, dtype=np.float32)
        ).transpose(0, 2, 3, 1).astype(self.compute_dtype)

    def _to_numpy(self, value: Any) -> np.ndarray:
        value = value.astype(self.mx.float32)
        self.mx.eval(value)
        return np.asarray(value, dtype=np.float32)

    def forward_trunk_features(self, input_nchw: np.ndarray) -> dict[str, np.ndarray]:
        with self._stream_ctx():
            _decoded, features = self.model.yolo(
                self._input_to_mx(input_nchw),
                stop_at=max(self.config.yolo.feature_indices),
            )
            outputs = {
                name: value.transpose(0, 3, 1, 2)
                for name, value in features.items()
            }
            outputs = {name: value.astype(self.mx.float32) for name, value in outputs.items()}
            self.mx.eval(*outputs.values())
        return {name: np.asarray(value, dtype=np.float32) for name, value in outputs.items()}

    def forward_detector_heads(self, trunk_features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the seg + DB heads on trunk features (NCHW numpy arrays)."""
        with self._stream_ctx():
            features = {
                index: self._feature_to_mx(trunk_features[f"trunk.feature_{index}"])
                for index in self.config.yolo.feature_indices
            }
            mask, lines = self.model.heads(features)
            outputs = {
                "head.mask": mask.transpose(0, 3, 1, 2),
                "head.lines": lines.transpose(0, 3, 1, 2),
            }
            outputs = {name: value.astype(self.mx.float32) for name, value in outputs.items()}
            self.mx.eval(*outputs.values())
        return {name: np.asarray(value, dtype=np.float32) for name, value in outputs.items()}

    def forward_yolo_decoded(self, input_nchw: np.ndarray) -> np.ndarray:
        with self._stream_ctx():
            decoded, _features = self.model.yolo(self._input_to_mx(input_nchw))
            if decoded is None:
                raise TextDetBackendUnavailable("MLX YOLO Detect layer did not produce output.")
            decoded = decoded.astype(self.mx.float32)
            self.mx.eval(decoded)
        return np.asarray(decoded, dtype=np.float32)

    def forward(self, input_nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return decoded YOLO blocks, segmentation ``mask``, and DB ``lines``."""
        with self._stream_ctx():
            yolo_decoded, mask, lines = self.model(self._input_to_mx(input_nchw))
            yolo_decoded = yolo_decoded.astype(self.mx.float32)
            mask = mask.transpose(0, 3, 1, 2).astype(self.mx.float32)
            lines = lines.transpose(0, 3, 1, 2).astype(self.mx.float32)
            self.mx.eval(yolo_decoded, mask, lines)
        return (
            np.asarray(yolo_decoded, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
            np.asarray(lines, dtype=np.float32),
        )
