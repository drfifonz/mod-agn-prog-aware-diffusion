"""Mixins for generative model utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

try:  # optional dependency
    from pytorch_lightning.loggers import WandbLogger
except ImportError:  # pragma: no cover - Lightning API fallback
    WandbLogger = None  # type: ignore
from torchvision import utils


def _matrix_sqrt_eig(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute matrix square root using eigendecomposition (GPU-friendly, avoids scipy/MKL)."""
    # Ensure symmetric
    A = (A + A.T) / 2
    # Add regularization for numerical stability
    A = A + eps * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    # Clamp negative eigenvalues (numerical errors)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    # Compute sqrt
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    sqrt_A = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    return sqrt_A


def compute_fid_from_stats(
    mu_real: torch.Tensor,
    sigma_real: torch.Tensor,
    mu_fake: torch.Tensor,
    sigma_fake: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute FID score from pre-computed statistics (pure PyTorch, GPU-friendly)."""
    diff = mu_real - mu_fake

    # Product of covariances
    covmean = sigma_real @ sigma_fake
    # Matrix square root using eigendecomposition
    covmean_sqrt = _matrix_sqrt_eig(covmean, eps=eps)

    # FID formula
    fid = torch.sum(diff**2) + torch.trace(sigma_real) + torch.trace(sigma_fake) - 2 * torch.trace(covmean_sqrt)
    return fid


class SampleVisualizationMixin:
    """Utility helpers for saving and logging sample grids."""

    sampling_grid_nrow: int
    save_dir: Path

    def _save_samples_grid(
        self,
        samples: torch.Tensor,
        *,
        step: int | None,
        prefix: str,
    ) -> Tuple[torch.Tensor, Path]:
        grid = utils.make_grid(samples, nrow=self.sampling_grid_nrow)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if step is None:
            filename = f"{prefix}.png"
        else:
            filename = f"{prefix}-step_{step:06d}.png"
        grid_path = self.save_dir / filename
        utils.save_image(grid, grid_path)
        return grid, grid_path


class FIDValidationMixin(ABC):
    """Mixin that encapsulates Frechet Inception Distance evaluation.

    Uses pure PyTorch implementation to avoid Intel MKL DGEBAL crashes with scipy.
    """

    def __init__(
        self,
        fid_num_samples: int,
        fid_feature: int = 2048,
        *,
        low_vram_mode: bool = False,
        clear_cuda_cache: bool = True,
        use_cpu_for_fid: bool = False,
    ) -> None:
        self.fid_num_samples = int(fid_num_samples)
        self._fid_feature_dim = fid_feature
        self._fid_low_vram_mode = bool(low_vram_mode)
        self._fid_clear_cuda_cache = bool(clear_cuda_cache)
        self._fid_use_cpu_for_fid = bool(use_cpu_for_fid)

        # Load Inception model for feature extraction
        self._inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self._inception.fc = torch.nn.Identity()  # Remove final FC layer to get 2048-dim features
        self._inception.eval()
        for param in self._inception.parameters():
            param.requires_grad = False

        # Storage for features
        self._real_features: list[torch.Tensor] = []
        self._fake_features: list[torch.Tensor] = []
        self._fid_real_count = 0
        self._fid_fake_count = 0
        self._fid_warned_shortfall = False

    def _log(self, message: str) -> None:
        logger = getattr(self, "logger", None)
        if logger is None or WandbLogger is None:
            self.print(message)
        elif isinstance(logger, WandbLogger):
            logger.log_text("fid_progress", columns=["status"], data=[[message]])
        else:
            self.print(message)

    def on_validation_epoch_start(self) -> None:
        # Skip FID setup during sanity checking
        trainer = getattr(self, "trainer", None)
        if trainer is not None and getattr(trainer, "sanity_checking", False):
            return

        self._real_features = []
        self._fake_features = []
        self._fid_real_count = 0
        self._fid_fake_count = 0
        self._fid_warned_shortfall = False
        self._inception.to(self._fid_device())
        self._log("[fid] validation epoch start: reset")
        if self._fid_low_vram_mode:
            self._maybe_clear_cuda_cache()

    def validation_step(self, batch, batch_idx: int) -> None:  # type: ignore[override]
        # Skip FID sample collection during sanity checking
        trainer = getattr(self, "trainer", None)
        if trainer is not None and getattr(trainer, "sanity_checking", False):
            return

        images, _ = batch
        self._update_fid_real(images)
        self._update_fid_fake(images.size(0))

    def on_validation_epoch_end(self) -> None:
        # Skip FID computation during sanity checking
        trainer = getattr(self, "trainer", None)
        if trainer is not None and getattr(trainer, "sanity_checking", False):
            self._log("[fid] skipping FID computation during sanity check")
            return

        if self._fid_real_count == 0 or self._fid_fake_count == 0:
            fid_value = torch.tensor(float("nan"), device=self.device)
            self.log("fid", fid_value, prog_bar=True, sync_dist=True)
            self._log("[fid] insufficient samples to compute metric")
            return

        min_samples = min(self._fid_real_count, self._fid_fake_count)
        if min_samples < self.fid_num_samples and not self._fid_warned_shortfall:
            self._log(
                "[fid] limited samples: "
                f"real={self._fid_real_count}, fake={self._fid_fake_count}, requested={self.fid_num_samples}"
            )
            self._fid_warned_shortfall = True

        try:
            # Compute FID using pure PyTorch (GPU-friendly, no scipy/MKL)
            real_features = torch.cat(self._real_features, dim=0).float()
            fake_features = torch.cat(self._fake_features, dim=0).float()

            # Compute statistics
            mu_real = real_features.mean(dim=0)
            mu_fake = fake_features.mean(dim=0)

            # Covariance (unbiased estimator)
            real_centered = real_features - mu_real
            fake_centered = fake_features - mu_fake
            n_real = real_features.shape[0]
            n_fake = fake_features.shape[0]
            sigma_real = (real_centered.T @ real_centered) / (n_real - 1)
            sigma_fake = (fake_centered.T @ fake_centered) / (n_fake - 1)

            # Compute FID
            fid_value = compute_fid_from_stats(mu_real, sigma_real, mu_fake, sigma_fake)

        except Exception as e:
            self._log(f"[fid] compute failed: {e}")
            fid_value = torch.tensor(float("nan"), device=self.device)

        self.log("fid", fid_value, prog_bar=True, sync_dist=True)
        self._log(f"[fid] computed value={fid_value.item():.4f}")

        # Clear feature storage
        self._real_features = []
        self._fake_features = []
        self._maybe_clear_cuda_cache()

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract Inception features from images."""
        self._inception.eval()
        features = self._inception(images)
        return features

    def _update_fid_real(self, images: torch.Tensor) -> None:
        remaining = self.fid_num_samples - self._fid_real_count
        if remaining <= 0:
            return
        take = min(images.size(0), remaining)
        real_chunk = images[:take]
        prepared = self._prepare_for_fid(
            real_chunk, already_normalized=getattr(self, "_real_images_in_unit_interval", False)
        )
        prepared = self._move_to_fid_device(prepared)

        with torch.no_grad():
            features = self._extract_features(prepared)
        self._real_features.append(features)
        self._fid_real_count += take
        self._log(f"[fid] collected real samples: {self._fid_real_count}/{self.fid_num_samples}")
        if self._fid_low_vram_mode:
            self._maybe_clear_cuda_cache()

    def _update_fid_fake(self, chunk_size: int) -> None:
        remaining = self.fid_num_samples - self._fid_fake_count
        if remaining <= 0:
            return
        take = min(chunk_size, remaining)
        samples = self.generate_samples_for_fid(take)
        samples = samples.to(self._fid_device())
        prepared = self._prepare_for_fid(samples, already_normalized=True)

        with torch.no_grad():
            features = self._extract_features(prepared)
        self._fake_features.append(features)
        self._fid_fake_count += take
        self._log(f"[fid] generated fake samples: {self._fid_fake_count}/{self.fid_num_samples}")
        if self._fid_low_vram_mode:
            self._maybe_clear_cuda_cache()

    @abstractmethod
    def generate_samples_for_fid(self, num_samples: int) -> torch.Tensor:
        """Subclasses must return denormalized samples in [0, 1]."""

    @staticmethod
    def _prepare_for_fid(images: torch.Tensor, *, already_normalized: bool) -> torch.Tensor:
        if not already_normalized:
            images = (images.clamp(-1, 1) + 1) / 2.0
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        # Normalize for Inception (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        return images

    def _fid_device(self) -> torch.device:
        if self._fid_use_cpu_for_fid:
            return torch.device("cpu")
        module_device = getattr(self, "device", torch.device("cpu"))
        if isinstance(module_device, torch.device):
            return module_device
        return torch.device(module_device)

    def _move_to_fid_device(self, tensor: torch.Tensor) -> torch.Tensor:
        target = self._fid_device()
        if tensor.device == target:
            return tensor
        return tensor.to(target)

    def _maybe_clear_cuda_cache(self) -> None:
        if not self._fid_clear_cuda_cache:
            return
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
