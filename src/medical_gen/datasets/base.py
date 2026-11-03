"""Base dataset config and datamodule abstractions."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterator, TypeVar

import pytorch_lightning as pl
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler


class InfiniteSampler(Sampler[int]):
    """Wraps a sampler to yield indices infinitely, preventing epoch increments."""

    def __init__(self, sampler: Sampler[int]) -> None:
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        while True:
            yield from self.sampler

    def __len__(self) -> int:
        # Return a very large number to prevent Lightning from computing epoch length
        return 2**62


class BaseDatasetConfig(BaseModel):
    """Common fields for dataset configs."""

    data_dir: str = "data"


ConfigT = TypeVar("ConfigT", bound=BaseDatasetConfig)


class BaseDataModule(pl.LightningDataModule, Generic[ConfigT], ABC):
    """Abstract data module tying together config and loaders."""

    def __init__(
        self,
        config: ConfigT,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        enable_validation: bool,
    ) -> None:
        super().__init__()
        if num_workers < 0:
            raise ValueError("'num_workers' must be non-negative.")
        self.config = config
        self.batch_size = batch_size
        # available_workers
        if (available_workers := os.cpu_count()) is not None:
            self.num_workers = min(num_workers, max(0, available_workers))
            print(os.cpu_count(), available_workers)
            print(
                f"Detected {available_workers} CPU cores. Using {self.num_workers} workers for data loading.in {self.__class__.__name__}."
            )
            # raise ValueError("'num_workers' must be non-negative.")
        else:
            self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.enable_validation = enable_validation
        self._train_dataset: Dataset[Any] | None = None
        self._val_dataset: Dataset[Any] | None = None
        self._train_steps_per_epoch: int | None = None

    def set_train_steps_per_epoch(self, steps: int | None) -> None:
        """Limit the number of unique batches per epoch by resampling with replacement."""

        if steps is not None and steps <= 0:
            raise ValueError("'steps' must be positive when provided.")
        self._train_steps_per_epoch = steps

    @abstractmethod
    def prepare_data(self) -> None:  # pragma: no cover - implementation specific
        """Prepare datasets (download, etc.)."""

    @abstractmethod
    def setup(self, stage: str | None = None) -> None:  # pragma: no cover
        """Create dataset objects for a given stage."""

    @abstractmethod
    def train_dataloader(self) -> DataLoader[Any]:  # pragma: no cover
        """Return dataloader for training."""

    def val_dataloader(self) -> DataLoader[Any] | list[DataLoader[Any]]:
        """Optional validation dataloader, overridable."""
        return []

    @property
    def has_validation(self) -> bool:
        """Return whether validation is enabled."""
        return bool(self.enable_validation)

    def _make_dataloader(
        self,
        dataset: Dataset[Any] | None,
        *,
        shuffle: bool,
        drop_last: bool = False,
        infinite: bool = False,
    ) -> DataLoader[Any]:
        if dataset is None:
            raise RuntimeError("Requested dataloader before dataset was initialized in setup().")
        sampler: Sampler[int] | None = None
        if shuffle and self._train_steps_per_epoch is not None:
            # Use infinite sampler to prevent epoch increments - max_steps controls termination
            base_sampler = RandomSampler(dataset, replacement=True)
            sampler = (
                InfiniteSampler(base_sampler)
                if infinite
                else RandomSampler(
                    dataset,
                    replacement=True,
                    num_samples=self._train_steps_per_epoch * self.batch_size,
                )
            )
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers // 2,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )
