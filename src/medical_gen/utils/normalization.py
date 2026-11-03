"""Normalization helpers for tensor data."""

from __future__ import annotations

from torch import Tensor


def normalize_to_neg_one_to_one(img: Tensor) -> Tensor:
    """Map tensors from [0, 1] into [-1, 1] range."""

    return img * 2 - 1
