"""Shared shape and normalization helpers."""
from .shapes import normalize_hw

def normalize_to_neg_one_to_one(images):
    from .normalization import normalize_to_neg_one_to_one as normalize
    return normalize(images)

__all__=['normalize_hw','normalize_to_neg_one_to_one']
