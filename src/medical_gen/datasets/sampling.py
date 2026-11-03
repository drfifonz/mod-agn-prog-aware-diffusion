"""Deterministic class quotas for published sampling protocols (no oversampling)."""
from collections import Counter
import math
import random
from .records import ImageRecord


def subsample_distribution(records: list[ImageRecord], classes: list[str],
                           distribution: dict[str, float], seed: int) -> list[ImageRecord]:
    if set(distribution) != set(classes) or any(not math.isfinite(v) or v <= 0 for v in distribution.values()):
        raise ValueError('Distribution must define a positive finite weight for each class')
    total = sum(distribution.values())
    weights = {c: distribution[c] / total for c in classes}
    by_class = {c: [r for r in records if r.label == c] for c in classes}
    if any(not by_class[c] for c in classes):
        raise ValueError('Distribution subsampling requires every class to be present')
    budget = int(min(len(by_class[c]) / weights[c] for c in classes))
    rng = random.Random(seed)
    result = []
    for c in classes:
        count = int(budget * weights[c])
        if count < 1:
            raise ValueError(f'Distribution gives zero images to {c}; increase the source pool')
        result.extend(rng.sample(by_class[c], min(count, len(by_class[c]))))
    return result


def select_training(real: list[ImageRecord], synthetic: dict[str, list[ImageRecord]],
                    classes: list[str], mode: str, stratified: bool, seed: int,
                    synthetic_per_class: int,
                    distribution: dict[str, float] | None = None) -> list[ImageRecord]:
    if mode not in {'real', 'synthetic', 'mixed'}:
        raise ValueError(f'Unknown training mode: {mode}')
    if len(classes) != len(set(classes)) or not classes:
        raise ValueError('Classes must be nonempty and unique')
    if any(r.source != 'real' or r.split != 'train' or r.label not in classes for r in real):
        raise ValueError('Real training input must contain real train records from configured classes')
    if synthetic_per_class < 0:
        raise ValueError('synthetic_per_class must be non-negative')
    if mode == 'real':
        if stratified:
            if distribution is None:
                raise ValueError('Real-only stratified sampling requires target_distribution')
            return subsample_distribution(real, classes, distribution, seed)
        return list(real)
    counts = Counter(r.label for r in real)
    if mode == 'mixed' and stratified:
        target = max(counts.values(), default=0)
        if target == 0:
            raise ValueError('Mixed training requires real training images')
        quotas = {c: target - counts[c] for c in classes}
    elif mode == 'synthetic' and stratified:
        if distribution is None:
            raise ValueError('Synthetic-only stratified sampling requires target_distribution')
        return subsample_distribution([r for c in classes for r in synthetic.get(c, [])], classes, distribution, seed)
    else:
        if synthetic_per_class < 1:
            raise ValueError('Equal-addition sampling requires synthetic_per_class > 0')
        quotas = dict.fromkeys(classes, synthetic_per_class)
    shortages = [f'{c}: need {n}, available {len(synthetic.get(c, []))}'
                 for c, n in quotas.items() if len(synthetic.get(c, [])) < n]
    if shortages:
        raise ValueError('Insufficient synthetic images: ' + '; '.join(shortages))
    rng = random.Random(seed)
    selected = list(real) if mode == 'mixed' else []
    seen = {r.path.resolve() for r in selected}
    for c in classes:
        candidates = synthetic.get(c, [])
        if any(r.label != c or r.source != 'synthetic' for r in candidates):
            raise ValueError(f'Invalid synthetic records for class {c}')
        for record in rng.sample(candidates, quotas[c]):
            key = record.path.resolve()
            if key in seen:
                raise ValueError(f'Duplicate selected image: {key.name}')
            seen.add(key)
            selected.append(record)
    return selected
