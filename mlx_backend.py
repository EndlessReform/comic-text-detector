from __future__ import annotations

import importlib
import json
import operator
from itertools import accumulate
from pathlib import Path
from typing import Any

import numpy as np

from comic_text_detector.backends import TextDetBackendUnavailable


CONFIG_FILENAME = "config.json"


def _autopad(kernel_size: int, padding: int | None = None) -> int:
    return kernel_size // 2 if padding is None else padding


def _scaled_depth(repeats: int, depth_multiple: float) -> int:
    return max(round(repeats * depth_multiple), 1) if repeats > 1 else repeats


def _non_overlapping_sliding_windows(
    mx: Any,
    x: Any,
    shape: tuple[int, ...],
    window_shape: list[int],
) -> Any:
    new_shape = [shape[0]]
    for size, window in zip(shape[1:], window_shape):
        new_shape.append(size // window)
        new_shape.append(window)
    new_shape.append(shape[-1])

    last_axis = len(new_shape) - 1
    axis_order = [0, *range(1, last_axis, 2), *range(2, last_axis, 2), last_axis]
    return x.reshape(new_shape).transpose(axis_order)


def _sliding_windows(
    mx: Any,
    x: Any,
    window_shape: list[int],
    window_strides: list[int],
) -> Any:
    spatial_dims = x.shape[1:-1]
    shape = x.shape
    if all(
        window == stride and size % window == 0
        for size, window, stride in zip(spatial_dims, window_shape, window_strides)
    ):
        return _non_overlapping_sliding_windows(mx, x, shape, window_shape)

    strides = list(reversed(list(accumulate(reversed(shape + (1,)), operator.mul))))[1:]
    final_shape = [shape[0]]
    final_shape += [
        (size - window) // stride + 1
        for size, window, stride in zip(spatial_dims, window_shape, window_strides)
    ]
    final_shape += window_shape
    final_shape += [shape[-1]]

    final_strides = strides[:1]
    final_strides += [
        original_stride * stride
        for original_stride, stride in zip(strides[1:-1], window_strides)
    ]
    final_strides += strides[1:-1]
    final_strides += strides[-1:]
    return mx.as_strided(x, final_shape, final_strides)


class MlxYoloTrunk:
    """YOLOv5 trunk subset used by the detector fixture.

    This intentionally stops at layer 9 and returns only the saved feature maps
    consumed by the later heads. Detect/FPN/head blocks are ported separately.
    """

    feature_indices = (1, 3, 5, 7, 9)

    def __init__(self, mx: Any, config: dict[str, Any], weights: dict[str, Any]):
        self.mx = mx
        self.config = config
        self.weights = weights
        self._fused_conv_cache: dict[str, tuple[Any, Any]] = {}
        self.cfg = config["yolov5"]["cfg"]
        self.depth_multiple = float(self.cfg["depth_multiple"])
        self.layers = self.cfg["backbone"]

    def __call__(self, input_nchw: np.ndarray) -> dict[str, np.ndarray]:
        if input_nchw.ndim != 4 or input_nchw.shape[1] != 3:
            raise ValueError(
                f"Expected NCHW input with 3 channels, got shape {input_nchw.shape}"
            )

        x = self.mx.array(input_nchw.astype(np.float32, copy=False)).transpose(
            0,
            2,
            3,
            1,
        )
        features: dict[str, Any] = {}

        for index, (from_index, repeats, module_name, args) in enumerate(self.layers):
            if from_index != -1:
                raise TextDetBackendUnavailable(
                    "MLX trunk only supports sequential backbone layers; "
                    f"layer {index} uses from={from_index}."
                )

            prefix = f"blk_det.weights.model.{index}"
            if module_name == "Conv":
                x = self._conv_block(
                    x,
                    prefix,
                    kernel_size=int(args[1]),
                    stride=int(args[2]),
                    padding=args[3] if len(args) > 3 else None,
                )
            elif module_name == "C3":
                x = self._c3(x, prefix, repeats=_scaled_depth(int(repeats), self.depth_multiple))
            elif module_name == "SPPF":
                x = self._sppf(x, prefix, kernel_size=int(args[1]))
            else:
                raise TextDetBackendUnavailable(
                    f"Unsupported MLX trunk layer type at index {index}: {module_name}"
                )

            if index in self.feature_indices:
                features[f"trunk.feature_{index}"] = x.transpose(0, 3, 1, 2)

        self.mx.eval(*features.values())
        return {name: np.asarray(value, dtype=np.float32) for name, value in features.items()}

    def _weight(self, key: str) -> Any:
        return self.weights[key]

    def _conv_block(
        self,
        x: Any,
        prefix: str,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        activation: str = "silu",
    ) -> Any:
        weight, bias = self._fused_conv_weight_bias(prefix)
        y = self.mx.conv2d(x, weight, stride, _autopad(kernel_size, padding), 1, groups)
        y = y + bias
        return self._activate(y, activation)

    def _fused_conv_weight_bias(self, prefix: str, eps: float = 1e-3) -> tuple[Any, Any]:
        if prefix in self._fused_conv_cache:
            return self._fused_conv_cache[prefix]

        conv_weight = self._weight(f"{prefix}.conv.weight")
        weight = self._weight(f"{prefix}.bn.weight")
        bias = self._weight(f"{prefix}.bn.bias")
        running_mean = self._weight(f"{prefix}.bn.running_mean")
        running_var = self._weight(f"{prefix}.bn.running_var")
        scale = weight * self.mx.rsqrt(running_var + eps)

        fused_weight = conv_weight * scale.reshape((-1, 1, 1, 1))
        fused_bias = bias - running_mean * scale
        self._fused_conv_cache[prefix] = (fused_weight, fused_bias)
        return fused_weight, fused_bias

    def _activate(self, x: Any, activation: str) -> Any:
        if activation == "silu":
            return x * self._sigmoid(x)
        if activation == "leaky":
            return self.mx.maximum(x, x * 0.1)
        if activation == "relu":
            return self.mx.maximum(x, 0)
        if activation == "identity":
            return x
        raise ValueError(f"Unsupported activation: {activation}")

    def _sigmoid(self, x: Any) -> Any:
        return 1 / (1 + self.mx.exp(-x))

    def _c3(self, x: Any, prefix: str, *, repeats: int, shortcut: bool = True) -> Any:
        y1 = self._conv_block(x, f"{prefix}.cv1", kernel_size=1)
        for index in range(repeats):
            y1 = self._bottleneck(y1, f"{prefix}.m.{index}", shortcut=shortcut)
        y2 = self._conv_block(x, f"{prefix}.cv2", kernel_size=1)
        return self._conv_block(
            self.mx.concatenate((y1, y2), axis=-1),
            f"{prefix}.cv3",
            kernel_size=1,
        )

    def _bottleneck(self, x: Any, prefix: str, *, shortcut: bool = True) -> Any:
        y = self._conv_block(x, f"{prefix}.cv1", kernel_size=1)
        y = self._conv_block(y, f"{prefix}.cv2", kernel_size=3)
        return x + y if shortcut and x.shape[-1] == y.shape[-1] else y

    def _sppf(self, x: Any, prefix: str, *, kernel_size: int) -> Any:
        x = self._conv_block(x, f"{prefix}.cv1", kernel_size=1)
        y1 = self._max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        y2 = self._max_pool2d(
            y1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        y3 = self._max_pool2d(
            y2,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        return self._conv_block(
            self.mx.concatenate((x, y1, y2, y3), axis=-1),
            f"{prefix}.cv2",
            kernel_size=1,
        )

    def _max_pool2d(self, x: Any, *, kernel_size: int, stride: int, padding: int) -> Any:
        if padding > 0:
            x = self.mx.pad(
                x,
                [(0, 0), (padding, padding), (padding, padding), (0, 0)],
                constant_values=-float("inf"),
            )
        windows = _sliding_windows(self.mx, x, [kernel_size, kernel_size], [stride, stride])
        return self.mx.max(windows, axis=(-3, -2))


class MlxTextDetComputeBackend:
    name = "mlx"

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).expanduser()
        self.mx = self._import_mlx()
        self.config_path, self.weights_path = self._resolve_artifact(self.model_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.weights = self.mx.load(str(self.weights_path))
        self.trunk = MlxYoloTrunk(self.mx, self.config, self.weights)

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
    def _resolve_artifact(model_path: Path) -> tuple[Path, Path]:
        if model_path.is_dir():
            config_path = model_path / CONFIG_FILENAME
            if not config_path.is_file():
                raise TextDetBackendUnavailable(
                    f"MLX detector artifact is missing {CONFIG_FILENAME}: {model_path}"
                )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            weights_path = model_path / config["artifact"]["file"]
        elif model_path.suffix == ".safetensors":
            weights_path = model_path
            config_path = model_path.with_name(CONFIG_FILENAME)
        else:
            raise TextDetBackendUnavailable(
                "The MLX detector backend requires a converted MLX artifact directory or .safetensors file. "
                "Run comic_text_detector.scripts.convert_to_mlx dump first."
            )

        if not weights_path.is_file():
            raise TextDetBackendUnavailable(f"MLX detector weights not found: {weights_path}")
        if not config_path.is_file():
            raise TextDetBackendUnavailable(
                f"MLX detector config not found: {config_path}"
            )
        return config_path, weights_path

    def forward_trunk_features(self, input_nchw: np.ndarray) -> dict[str, np.ndarray]:
        return self.trunk(input_nchw)

    def forward(self, input_nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise TextDetBackendUnavailable(
            "The MLX detector backend currently implements trunk feature parity only; "
            "segmentation and DB heads are not implemented yet."
        )
