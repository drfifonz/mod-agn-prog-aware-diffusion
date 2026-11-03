"""Explicit image records and patient-disjoint real-data manifests."""
from __future__ import annotations
from collections import defaultdict
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import random
from typing import Iterable

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    patient_id: str
    split: str
    source: str = 'real'


def validate_partitions(records: Iterable[ImageRecord]) -> None:
    patients: dict[str, str] = {}
    images: set[Path] = set()
    for record in records:
        path = record.path.resolve()
        if path in images:
            raise ValueError(f'Duplicate image or image overlap: {path.name}')
        images.add(path)
        if record.source != 'real':
            continue
        if not record.patient_id.strip():
            raise ValueError('Every real image requires a patient_id')
        if record.split not in {'', 'train', 'val', 'test'}:
            raise ValueError(f'Invalid split: {record.split}')
        previous = patients.setdefault(record.patient_id, record.split)
        if previous != record.split:
            raise ValueError(f'Cross-split patient overlap: {record.patient_id}')


def load_real_manifest(path: Path, root: Path, classes: list[str]) -> list[ImageRecord]:
    path, root = path.expanduser().resolve(), root.expanduser().resolve()
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'path', 'label', 'patient_id', 'split'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest requires columns: {sorted(required)}')
        records = []
        for row in reader:
            if not row['path'].strip():
                raise ValueError('Image path must not be empty')
            image = (root / Path(row['path']).expanduser()).resolve()
            if row['label'] not in classes:
                raise ValueError(f'Unknown class: {row["label"]}')
            if not image.is_file():
                raise FileNotFoundError(f'Image not found: {image}')
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f'Unsupported image extension: {image.name}')
            records.append(ImageRecord(image, row['label'], row['patient_id'].strip(), row['split'].strip()))
    if not records:
        raise ValueError('Manifest is empty')
    validate_partitions(records)
    assigned = {bool(r.split) for r in records}
    if len(assigned) != 1:
        raise ValueError('Manifest must have either all splits assigned or all splits empty')
    return records


def write_manifest(records: Iterable[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'label', 'patient_id', 'split', 'source'])
        writer.writeheader()
        for record in records:
            writer.writerow({'path': str(record.path), 'label': record.label, 'patient_id': record.patient_id,
                             'split': record.split, 'source': record.source})


def split_by_patient(records: list[ImageRecord], seed: int, train_fraction: float = .8,
                     val_fraction: float = .1) -> list[ImageRecord]:
    validate_partitions(records)
    if any(r.split for r in records):
        raise ValueError('Patient splitting expects unassigned input splits')
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError('Train and validation fractions must be positive and sum to less than one')
    patients = sorted({r.patient_id for r in records})
    if len(patients) < 3:
        raise ValueError('At least three patients are required')
    random.Random(seed).shuffle(patients)
    n_train = int(len(patients) * train_fraction)
    n_val = int(len(patients) * val_fraction)
    if min(n_train, n_val, len(patients)-n_train-n_val) < 1:
        raise ValueError('Requested fractions create an empty patient subset; supply more patients or explicit splits')
    mapping = {p: ('train' if i < n_train else 'val' if i < n_train+n_val else 'test')
               for i, p in enumerate(patients)}
    result = [replace(r, split=mapping[r.patient_id]) for r in records]
    validate_partitions(result)
    return result


def generated_pool(root: Path, classes: list[str]) -> dict[str, list[ImageRecord]]:
    """Load only individual sample_ images; montage grids are never observations."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Synthetic root not found: {root}')
    return {name: [ImageRecord(p.resolve(), name, '', 'train', 'synthetic')
                   for p in sorted((root / name).glob('sample_*'))
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            for name in classes}
