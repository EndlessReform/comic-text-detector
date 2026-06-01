from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_FILENAME = "config.json"


@dataclass(frozen=True)
class WeightsConfig:
    file: str
    format: str = "safetensors"
    layout: str = "mlx-ohwi"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeightsConfig":
        return cls(
            file=str(data["file"]),
            format=str(data.get("format", "safetensors")),
            layout=str(data.get("layout", "mlx-ohwi")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "format": self.format, "layout": self.layout}


@dataclass(frozen=True)
class FusionConfig:
    conv_bn: str = "forward"
    trunk_bn_eps: float = 1e-3
    head_bn_eps: float = 1e-5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FusionConfig":
        return cls(
            conv_bn=str(data.get("conv_bn", "forward")),
            trunk_bn_eps=float(data.get("trunk_bn_eps", 1e-3)),
            head_bn_eps=float(data.get("head_bn_eps", 1e-5)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conv_bn": self.conv_bn,
            "trunk_bn_eps": self.trunk_bn_eps,
            "head_bn_eps": self.head_bn_eps,
        }


@dataclass(frozen=True)
class YoloLayerConfig:
    type: str
    from_index: int | list[int]
    repeats: int = 1
    out_channels: int | None = None
    in_channels: int | list[int] | None = None
    kernel: int | None = None
    stride: int | None = None
    padding: int | None = None
    shortcut: bool = True
    scale_factor: int | None = None
    mode: str | None = None
    dimension: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "YoloLayerConfig":
        return cls(
            type=str(data["type"]),
            from_index=data["from"],
            repeats=int(data.get("repeats", 1)),
            out_channels=data.get("out_channels"),
            in_channels=data.get("in_channels"),
            kernel=data.get("kernel"),
            stride=data.get("stride"),
            padding=data.get("padding"),
            shortcut=bool(data.get("shortcut", True)),
            scale_factor=data.get("scale_factor"),
            mode=data.get("mode"),
            dimension=data.get("dimension"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "from": self.from_index,
            "repeats": self.repeats,
        }
        for key in (
            "in_channels",
            "out_channels",
            "kernel",
            "stride",
            "padding",
            "scale_factor",
            "mode",
            "dimension",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.type == "C3":
            data["shortcut"] = self.shortcut
        return data


@dataclass(frozen=True)
class YoloConfig:
    depth_multiple: float
    width_multiple: float
    feature_indices: tuple[int, ...]
    detect_indices: tuple[int, ...]
    anchors: tuple[tuple[tuple[int, int], ...], ...]
    layers: tuple[YoloLayerConfig, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "YoloConfig":
        return cls(
            depth_multiple=float(data["depth_multiple"]),
            width_multiple=float(data["width_multiple"]),
            feature_indices=tuple(int(value) for value in data["feature_indices"]),
            detect_indices=tuple(int(value) for value in data["detect_indices"]),
            anchors=tuple(
                tuple((int(pair[0]), int(pair[1])) for pair in layer)
                for layer in data["anchors"]
            ),
            layers=tuple(YoloLayerConfig.from_dict(layer) for layer in data["layers"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_multiple": self.depth_multiple,
            "width_multiple": self.width_multiple,
            "feature_indices": list(self.feature_indices),
            "detect_indices": list(self.detect_indices),
            "anchors": [[list(pair) for pair in layer] for layer in self.anchors],
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass(frozen=True)
class HeadConfig:
    activation: str = "leaky"
    layers: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HeadConfig":
        data = data or {}
        return cls(
            activation=str(data.get("activation", "leaky")),
            layers=tuple(dict(layer) for layer in data.get("layers", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"activation": self.activation, "layers": [dict(layer) for layer in self.layers]}


@dataclass(frozen=True)
class HeadsConfig:
    segmentation: HeadConfig = field(default_factory=HeadConfig)
    db: HeadConfig = field(default_factory=HeadConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadsConfig":
        return cls(
            segmentation=HeadConfig.from_dict(data.get("segmentation")),
            db=HeadConfig.from_dict(data.get("db")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segmentation": self.segmentation.to_dict(),
            "db": self.db.to_dict(),
        }


@dataclass(frozen=True)
class TextDetConfig:
    model_type: str
    architectures: tuple[str, ...]
    format_version: int
    torch_dtype: str
    input_layout: str
    internal_layout: str
    image_size: int
    num_classes: int
    id2label: dict[str, str]
    label2id: dict[str, int]
    weights: WeightsConfig
    fusion: FusionConfig
    yolo: YoloConfig
    heads: HeadsConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextDetConfig":
        if data.get("model_type") != "comic_text_detector":
            raise ValueError("Expected comic_text_detector MLX config.json")
        return cls(
            model_type=str(data["model_type"]),
            architectures=tuple(str(item) for item in data["architectures"]),
            format_version=int(data["format_version"]),
            torch_dtype=str(data["torch_dtype"]),
            input_layout=str(data["input_layout"]),
            internal_layout=str(data["internal_layout"]),
            image_size=int(data["image_size"]),
            num_classes=int(data["num_classes"]),
            id2label={str(key): str(value) for key, value in data["id2label"].items()},
            label2id={str(key): int(value) for key, value in data["label2id"].items()},
            weights=WeightsConfig.from_dict(data["weights"]),
            fusion=FusionConfig.from_dict(data["fusion"]),
            yolo=YoloConfig.from_dict(data["yolo"]),
            heads=HeadsConfig.from_dict(data["heads"]),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TextDetConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "format_version": self.format_version,
            "torch_dtype": self.torch_dtype,
            "input_layout": self.input_layout,
            "internal_layout": self.internal_layout,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "id2label": dict(self.id2label),
            "label2id": dict(self.label2id),
            "weights": self.weights.to_dict(),
            "fusion": self.fusion.to_dict(),
            "yolo": self.yolo.to_dict(),
            "heads": self.heads.to_dict(),
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / CONFIG_FILENAME).write_text(self.to_json_string(), encoding="utf-8")
