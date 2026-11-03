"""Shape utilities for image height/width handling."""

from __future__ import annotations

from typing import Iterable, Tuple


def _to_int(value: int | float) -> int:
    if isinstance(value, bool):  # prevent bool sneaking in
        raise TypeError("Boolean values are not valid size components.")
    return int(value)


def normalize_hw(
    size: int | Iterable[int],
    *,
    field_name: str = "size",
) -> Tuple[int, int]:
    """Normalize size arguments to a (height, width) tuple."""

    if isinstance(size, int):
        height = width = size
    else:
        try:
            first, second = size  # type: ignore[misc]
        except Exception as err:
            raise TypeError(
                f"{field_name} must be an int or iterable of two ints, got {size!r}."
            ) from err
        height, width = _to_int(first), _to_int(second)
    if height <= 0 or width <= 0:
        raise ValueError(f"{field_name} height/width must be positive, got {(height, width)}.")
    return int(height), int(width)
