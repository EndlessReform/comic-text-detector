from __future__ import annotations

import operator
from itertools import accumulate
from typing import Any


def autopad(kernel_size: int, padding: int | None = None) -> int:
    return kernel_size // 2 if padding is None else padding


def non_overlapping_sliding_windows(
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


def sliding_windows(
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
        return non_overlapping_sliding_windows(mx, x, shape, window_shape)

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
