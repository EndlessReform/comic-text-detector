from __future__ import annotations

from typing import Any

try:
    import mlx.nn as nn
except ModuleNotFoundError:
    nn = None

from comic_text_detector.backends import TextDetBackendUnavailable
from comic_text_detector.mlx_backend.configuration_textdet import TextDetConfig, YoloLayerConfig
from comic_text_detector.mlx_backend.ops import autopad, sliding_windows

ModuleBase = object if nn is None else nn.Module


def sigmoid(mx: Any, x: Any) -> Any:
    return 1 / (1 + mx.exp(-x))


def activate(mx: Any, x: Any, activation: str) -> Any:
    if activation == "silu":
        return x * sigmoid(mx, x)
    if activation == "leaky":
        return mx.maximum(x, x * 0.1)
    if activation == "relu":
        return mx.maximum(x, 0)
    if activation == "identity":
        return x
    raise ValueError(f"Unsupported activation: {activation}")


class MlxModule(ModuleBase):
    def __init__(self, mx: Any):
        super().__init__()
        self.mx = mx
        self._compiled_forward = None
        self._compile_failed = False

    def compile_shapeless(self) -> None:
        self._compiled_forward = self.mx.compile(self._forward, shapeless=True)

    def _forward_or_compiled(self, x: Any) -> Any:
        if self._compiled_forward is None:
            return self._forward(x)
        try:
            return self._compiled_forward(x)
        except ValueError:
            self._compiled_forward = None
            self._compile_failed = True
            return self._forward(x)


class MlxConv2d(MlxModule):
    def __init__(
        self,
        mx: Any,
        weights: dict[str, Any],
        prefix: str,
        *,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ):
        super().__init__(mx)
        self.weight = weights[f"{prefix}.weight"]
        self.bias = weights[f"{prefix}.bias"] if bias else None
        self.stride = stride
        self.padding = autopad(kernel_size, padding)

    def _forward(self, x: Any) -> Any:
        y = self.mx.conv2d(x, self.weight, self.stride, self.padding, 1, 1)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __call__(self, x: Any) -> Any:
        return self._forward_or_compiled(x)


class MlxConvTranspose2d(MlxModule):
    def __init__(
        self,
        mx: Any,
        weights: dict[str, Any],
        prefix: str,
        *,
        stride: int,
        padding: int,
        bias: bool = False,
    ):
        super().__init__(mx)
        self.weight = weights[f"{prefix}.weight"]
        self.bias = weights[f"{prefix}.bias"] if bias else None
        self.stride = stride
        self.padding = padding

    def _forward(self, x: Any) -> Any:
        y = self.mx.conv_transpose2d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __call__(self, x: Any) -> Any:
        return self._forward_or_compiled(x)


class MlxBatchNorm2d(MlxModule):
    def __init__(self, mx: Any, weights: dict[str, Any], prefix: str, *, eps: float):
        super().__init__(mx)
        self.weight = weights[f"{prefix}.weight"]
        self.bias = weights[f"{prefix}.bias"]
        self.running_mean = weights[f"{prefix}.running_mean"]
        self.running_var = weights[f"{prefix}.running_var"]
        self.eps = eps

    def __call__(self, x: Any) -> Any:
        return (
            (x - self.running_mean)
            * self.mx.rsqrt(self.running_var + self.eps)
            * self.weight
            + self.bias
        )


class MlxFusedConvBnAct(MlxModule):
    def __init__(
        self,
        mx: Any,
        weights: dict[str, Any],
        prefix: str,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        activation: str = "silu",
        eps: float = 1e-3,
    ):
        super().__init__(mx)
        conv_weight = weights[f"{prefix}.conv.weight"]
        weight = weights[f"{prefix}.bn.weight"]
        bias = weights[f"{prefix}.bn.bias"]
        running_mean = weights[f"{prefix}.bn.running_mean"]
        running_var = weights[f"{prefix}.bn.running_var"]
        scale = weight * mx.rsqrt(running_var + eps)
        self.weight = conv_weight * scale.reshape((-1, 1, 1, 1))
        self.bias = bias - running_mean * scale
        self.stride = stride
        self.padding = autopad(kernel_size, padding)
        self.groups = groups
        self.activation = activation

    def _forward(self, x: Any) -> Any:
        y = self.mx.conv2d(x, self.weight, self.stride, self.padding, 1, self.groups)
        return activate(self.mx, y + self.bias, self.activation)

    def __call__(self, x: Any) -> Any:
        return self._forward_or_compiled(x)


class MlxBottleneck(MlxModule):
    def __init__(
        self,
        mx: Any,
        weights: dict[str, Any],
        prefix: str,
        *,
        shortcut: bool,
        activation: str,
        eps: float,
    ):
        super().__init__(mx)
        self.cv1 = MlxFusedConvBnAct(
            mx, weights, f"{prefix}.cv1", kernel_size=1, activation=activation, eps=eps
        )
        self.cv2 = MlxFusedConvBnAct(
            mx, weights, f"{prefix}.cv2", kernel_size=3, activation=activation, eps=eps
        )
        self.shortcut = shortcut

    def __call__(self, x: Any) -> Any:
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut and x.shape[-1] == y.shape[-1] else y


class MlxC3(MlxModule):
    def __init__(
        self,
        mx: Any,
        weights: dict[str, Any],
        prefix: str,
        *,
        repeats: int,
        shortcut: bool = True,
        activation: str = "silu",
        eps: float = 1e-3,
    ):
        super().__init__(mx)
        self.cv1 = MlxFusedConvBnAct(
            mx, weights, f"{prefix}.cv1", kernel_size=1, activation=activation, eps=eps
        )
        self.blocks = [
            MlxBottleneck(
                mx,
                weights,
                f"{prefix}.m.{index}",
                shortcut=shortcut,
                activation=activation,
                eps=eps,
            )
            for index in range(repeats)
        ]
        self.cv2 = MlxFusedConvBnAct(
            mx, weights, f"{prefix}.cv2", kernel_size=1, activation=activation, eps=eps
        )
        self.cv3 = MlxFusedConvBnAct(
            mx, weights, f"{prefix}.cv3", kernel_size=1, activation=activation, eps=eps
        )

    def __call__(self, x: Any) -> Any:
        y1 = self.cv1(x)
        for block in self.blocks:
            y1 = block(y1)
        y2 = self.cv2(x)
        return self.cv3(self.mx.concatenate((y1, y2), axis=-1))


class MlxSPPF(MlxModule):
    def __init__(
        self,
        mx: Any,
        weights: dict[str, Any],
        prefix: str,
        *,
        kernel_size: int,
        eps: float,
    ):
        super().__init__(mx)
        self.cv1 = MlxFusedConvBnAct(mx, weights, f"{prefix}.cv1", kernel_size=1, eps=eps)
        self.cv2 = MlxFusedConvBnAct(mx, weights, f"{prefix}.cv2", kernel_size=1, eps=eps)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def __call__(self, x: Any) -> Any:
        x = self.cv1(x)
        y1 = self._max_pool2d(x)
        y2 = self._max_pool2d(y1)
        y3 = self._max_pool2d(y2)
        return self.cv2(self.mx.concatenate((x, y1, y2, y3), axis=-1))

    def _max_pool2d(self, x: Any) -> Any:
        if self.padding > 0:
            x = self.mx.pad(
                x,
                [(0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)],
                constant_values=-float("inf"),
            )
        windows = sliding_windows(
            self.mx,
            x,
            [self.kernel_size, self.kernel_size],
            [1, 1],
        )
        return self.mx.max(windows, axis=(-3, -2))


class MlxUpsampleNearest(MlxModule):
    def __init__(self, mx: Any, *, scale_factor: int):
        super().__init__(mx)
        self.scale_factor = scale_factor

    def __call__(self, x: Any) -> Any:
        x = self.mx.repeat(x, self.scale_factor, axis=1)
        return self.mx.repeat(x, self.scale_factor, axis=2)


class MlxConcat(MlxModule):
    def __call__(self, xs: list[Any]) -> Any:
        return self.mx.concatenate(xs, axis=-1)


class MlxYoloDetect(MlxModule):
    def __init__(
        self,
        mx: Any,
        config: TextDetConfig,
        weights: dict[str, Any],
        prefix: str,
    ):
        super().__init__(mx)
        self.num_classes = config.num_classes
        self.num_outputs = self.num_classes + 5
        self.anchors = weights[f"{prefix}.anchors"]
        self.convs = [
            MlxConv2d(mx, weights, f"{prefix}.m.{index}", bias=True)
            for index in range(len(config.yolo.detect_indices))
        ]

    def __call__(self, xs: list[Any], *, input_hw: tuple[int, int]) -> Any:
        decoded_per_layer = []
        na = int(self.anchors.shape[1])

        for index, x in enumerate(xs):
            raw = self.convs[index](x)
            batch, ny, nx, _channels = raw.shape
            raw = raw.reshape((batch, ny, nx, na, self.num_outputs)).transpose(0, 3, 1, 2, 4)
            y = sigmoid(self.mx, raw)

            stride_y = input_hw[0] / ny
            stride_x = input_hw[1] / nx
            if abs(stride_x - stride_y) > 1e-6:
                raise TextDetBackendUnavailable(
                    "MLX YOLO Detect currently expects square detector strides; "
                    f"got stride_x={stride_x}, stride_y={stride_y}."
                )
            stride = float(stride_y)
            grid = self._make_grid(nx, ny, na)
            anchor_grid = self.anchors[index].reshape((1, na, 1, 1, 2)) * stride

            xy = (y[..., 0:2] * 2 - 0.5 + grid) * stride
            wh = (y[..., 2:4] * 2) ** 2 * anchor_grid
            decoded = self.mx.concatenate((xy, wh, y[..., 4:]), axis=-1)
            decoded_per_layer.append(decoded.reshape((batch, -1, self.num_outputs)))

        return self.mx.concatenate(decoded_per_layer, axis=1)

    def _make_grid(self, nx: int, ny: int, na: int) -> Any:
        xv = self.mx.arange(nx, dtype=self.mx.float32)
        yv = self.mx.arange(ny, dtype=self.mx.float32)
        grid_x, grid_y = self.mx.meshgrid(xv, yv, indexing="xy")
        grid = self.mx.stack((grid_x, grid_y), axis=-1).reshape((1, 1, ny, nx, 2))
        return self.mx.broadcast_to(grid, (1, na, ny, nx, 2))


class MlxYoloLayer(MlxModule):
    def __init__(
        self,
        mx: Any,
        config: TextDetConfig,
        weights: dict[str, Any],
        layer: YoloLayerConfig,
        *,
        index: int,
    ):
        super().__init__(mx)
        self.from_index = layer.from_index
        self.layer_type = layer.type
        prefix = f"blk_det.weights.model.{index}"
        eps = config.fusion.trunk_bn_eps

        if layer.type == "Conv":
            self.module = MlxFusedConvBnAct(
                mx,
                weights,
                prefix,
                kernel_size=int(layer.kernel),
                stride=int(layer.stride or 1),
                padding=layer.padding,
                eps=eps,
            )
        elif layer.type == "C3":
            self.module = MlxC3(
                mx,
                weights,
                prefix,
                repeats=layer.repeats,
                shortcut=layer.shortcut,
                activation="silu",
                eps=eps,
            )
        elif layer.type == "SPPF":
            self.module = MlxSPPF(mx, weights, prefix, kernel_size=int(layer.kernel), eps=eps)
        elif layer.type == "Upsample":
            self.module = MlxUpsampleNearest(mx, scale_factor=int(layer.scale_factor))
        elif layer.type == "Concat":
            self.module = MlxConcat(mx)
        elif layer.type == "Detect":
            self.module = MlxYoloDetect(mx, config, weights, prefix)
        else:
            raise TextDetBackendUnavailable(
                f"Unsupported MLX YOLO layer type at index {index}: {layer.type}"
            )

    def __call__(self, x: Any, *, input_hw: tuple[int, int] | None = None) -> Any:
        if self.layer_type == "Detect":
            if input_hw is None:
                raise ValueError("Detect layer requires input_hw.")
            return self.module(x, input_hw=input_hw)
        return self.module(x)


class MlxYoloBlockDetector(MlxModule):
    def __init__(self, mx: Any, config: TextDetConfig, weights: dict[str, Any]):
        super().__init__(mx)
        self.feature_indices = config.yolo.feature_indices
        self.detect_index = len(config.yolo.layers) - 1
        self.layers = [
            MlxYoloLayer(mx, config, weights, layer, index=index)
            for index, layer in enumerate(config.yolo.layers)
        ]

    def __call__(
        self,
        input_nhwc: Any,
        *,
        stop_at: int | None = None,
    ) -> tuple[Any | None, dict[str, Any]]:
        input_hw = (int(input_nhwc.shape[1]), int(input_nhwc.shape[2]))
        x = input_nhwc
        outputs: list[Any] = []
        features: dict[str, Any] = {}
        decoded = None

        last_index = self.detect_index if stop_at is None else stop_at
        for index, layer in enumerate(self.layers):
            if index > last_index:
                break

            layer_input = self._layer_input(x, outputs, layer.from_index)
            x = layer(layer_input, input_hw=input_hw) if layer.layer_type == "Detect" else layer(layer_input)
            if layer.layer_type == "Detect":
                decoded = x
            outputs.append(x)
            if index in self.feature_indices:
                features[f"trunk.feature_{index}"] = x

        return decoded, features

    def _layer_input(self, x: Any, outputs: list[Any], from_index: int | list[int]) -> Any:
        if from_index == -1:
            return x
        if isinstance(from_index, int):
            return outputs[from_index]
        return [x if index == -1 else outputs[index] for index in from_index]


class MlxDownC3(MlxModule):
    def __init__(self, mx: Any, weights: dict[str, Any], prefix: str, *, eps: float, activation: str):
        super().__init__(mx)
        self.c3 = MlxC3(
            mx,
            weights,
            prefix,
            repeats=1,
            shortcut=True,
            activation=activation,
            eps=eps,
        )

    def __call__(self, x: Any) -> Any:
        windows = sliding_windows(self.mx, x, [2, 2], [2, 2])
        return self.c3(self.mx.mean(windows, axis=(-3, -2)))


class MlxUpC3(MlxModule):
    def __init__(self, mx: Any, weights: dict[str, Any], prefix: str, *, eps: float, activation: str):
        super().__init__(mx)
        self.c3 = MlxC3(
            mx,
            weights,
            f"{prefix}.0",
            repeats=1,
            shortcut=True,
            activation=activation,
            eps=eps,
        )
        self.up = MlxConvTranspose2d(mx, weights, f"{prefix}.1", stride=2, padding=1, bias=False)
        self.bn = MlxBatchNorm2d(mx, weights, f"{prefix}.2", eps=eps)

    def __call__(self, x: Any) -> Any:
        return activate(self.mx, self.bn(self.up(self.c3(x))), "relu")


class MlxSegmentationHead(MlxModule):
    def __init__(self, mx: Any, config: TextDetConfig, weights: dict[str, Any]):
        super().__init__(mx)
        eps = config.fusion.head_bn_eps
        activation = config.heads.segmentation.activation
        self.feature_indices = config.yolo.feature_indices
        self.down_conv1 = MlxDownC3(
            mx, weights, "text_seg.down_conv1.conv", eps=eps, activation=activation
        )
        self.upconv0 = MlxUpC3(mx, weights, "text_seg.upconv0.conv", eps=eps, activation=activation)
        self.upconv2 = MlxUpC3(mx, weights, "text_seg.upconv2.conv", eps=eps, activation=activation)
        self.upconv3 = MlxUpC3(mx, weights, "text_seg.upconv3.conv", eps=eps, activation=activation)
        self.upconv4 = MlxUpC3(mx, weights, "text_seg.upconv4.conv", eps=eps, activation=activation)
        self.upconv5 = MlxUpC3(mx, weights, "text_seg.upconv5.conv", eps=eps, activation=activation)
        self.upconv6 = MlxConvTranspose2d(
            mx, weights, "text_seg.upconv6.0", stride=2, padding=1, bias=False
        )

    def __call__(self, feats: dict[int, Any]) -> tuple[Any, list[Any]]:
        f160, f80, f40, f20, f3 = (feats[i] for i in self.feature_indices)
        cat = self.mx.concatenate

        d10 = self.down_conv1(f3)
        u20 = self.upconv0(d10)
        u40 = self.upconv2(cat((f20, u20), axis=-1))
        u80 = self.upconv3(cat((f40, u40), axis=-1))
        u160 = self.upconv4(cat((f80, u80), axis=-1))
        u320 = self.upconv5(cat((f160, u160), axis=-1))
        mask = sigmoid(self.mx, self.upconv6(u320))
        return mask, [f80, f40, u40]


class MlxDbBranch(MlxModule):
    def __init__(
        self,
        mx: Any,
        weights: dict[str, Any],
        prefix: str,
        *,
        eps: float,
        first_bias: bool,
        sigmoid_output: bool,
    ):
        super().__init__(mx)
        self.conv0 = MlxConv2d(
            mx, weights, f"{prefix}.0", kernel_size=3, padding=1, bias=first_bias
        )
        self.bn1 = MlxBatchNorm2d(mx, weights, f"{prefix}.1", eps=eps)
        self.up3 = MlxConvTranspose2d(mx, weights, f"{prefix}.3", stride=2, padding=0, bias=True)
        self.bn4 = MlxBatchNorm2d(mx, weights, f"{prefix}.4", eps=eps)
        self.up6 = MlxConvTranspose2d(mx, weights, f"{prefix}.6", stride=2, padding=0, bias=True)
        self.sigmoid_output = sigmoid_output

    def __call__(self, x: Any) -> Any:
        x = activate(self.mx, self.bn1(self.conv0(x)), "relu")
        x = activate(self.mx, self.bn4(self.up3(x)), "relu")
        x = self.up6(x)
        return sigmoid(self.mx, x) if self.sigmoid_output else x


class MlxDbHead(MlxModule):
    def __init__(self, mx: Any, config: TextDetConfig, weights: dict[str, Any]):
        super().__init__(mx)
        eps = config.fusion.head_bn_eps
        activation = config.heads.segmentation.activation
        self.upconv3 = MlxUpC3(mx, weights, "text_det.upconv3.conv", eps=eps, activation=activation)
        self.upconv4 = MlxUpC3(mx, weights, "text_det.upconv4.conv", eps=eps, activation=activation)
        self.projection = MlxConv2d(mx, weights, "text_det.conv.0", kernel_size=1, bias=True)
        self.projection_bn = MlxBatchNorm2d(mx, weights, "text_det.conv.1", eps=eps)
        self.threshold = MlxDbBranch(
            mx,
            weights,
            "text_det.thresh",
            eps=eps,
            first_bias=False,
            sigmoid_output=True,
        )
        self.binarize = MlxDbBranch(
            mx,
            weights,
            "text_det.binarize",
            eps=eps,
            first_bias=True,
            sigmoid_output=False,
        )

    def __call__(self, seg_features: list[Any]) -> Any:
        f80, f40, u40 = seg_features
        cat = self.mx.concatenate

        u80 = self.upconv3(cat((f40, u40), axis=-1))
        x = self.upconv4(cat((f80, u80), axis=-1))
        x = activate(self.mx, self.projection_bn(self.projection(x)), "relu")

        threshold_maps = self.threshold(x)
        shrink_maps = sigmoid(self.mx, self.binarize(x))
        return cat((shrink_maps, threshold_maps), axis=-1)


class MlxTextDetHeads(MlxModule):
    def __init__(self, mx: Any, config: TextDetConfig, weights: dict[str, Any]):
        super().__init__(mx)
        self.feature_indices = config.yolo.feature_indices
        self.segmentation = MlxSegmentationHead(mx, config, weights)
        self.db = MlxDbHead(mx, config, weights)

    def __call__(self, trunk_features: dict[int, Any]) -> tuple[Any, Any]:
        mask, seg_features = self.segmentation(trunk_features)
        lines = self.db(seg_features)
        return mask, lines


class MlxComicTextDetector(MlxModule):
    def __init__(self, mx: Any, config: TextDetConfig, weights: dict[str, Any]):
        super().__init__(mx)
        self.yolo = MlxYoloBlockDetector(mx, config, weights)
        self.heads = MlxTextDetHeads(mx, config, weights)

    def __call__(self, input_nhwc: Any) -> tuple[Any, Any, Any]:
        decoded, trunk_features = self.yolo(input_nhwc)
        if decoded is None:
            raise TextDetBackendUnavailable("MLX YOLO Detect layer did not produce output.")
        feature_by_index = {
            int(key.rsplit("_", 1)[1]): value for key, value in trunk_features.items()
        }
        mask, lines = self.heads(feature_by_index)
        return decoded, mask, lines

    def compile_conv_blocks(self) -> int:
        count = 0
        for module in self.modules():
            if isinstance(module, (MlxConv2d, MlxConvTranspose2d, MlxFusedConvBnAct)):
                module.compile_shapeless()
                count += 1
        return count


MlxYoloTrunk = MlxYoloBlockDetector
