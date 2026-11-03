from __future__ import annotations

from pathlib import Path
from random import random
from typing import Any, Literal, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from pytorch_lightning.loggers import WandbLogger

from ..utils import normalize_hw
from .ddpm_lib.denoising_diffusion_pytorch import (
    GaussianDiffusion as BaseGaussianDiffusion,
)
from .ddpm_lib.denoising_diffusion_pytorch import Unet, default, extract
from .mixins import FIDValidationMixin, SampleVisualizationMixin


class GaussianDiffusion(BaseGaussianDiffusion):
    """Gaussian diffusion with tuple-friendly image size and configurable loss type."""

    def __init__(
        self,
        *args: Any,
        image_size: tuple[int, int] | int,
        loss_type: Literal["l1", "l2"] = "l2",
        **kwargs: Any,
    ) -> None:
        self.loss_type = loss_type
        super().__init__(*args, image_size=image_size, **kwargs)

    def _normalize_if_needed(self, img: torch.Tensor) -> torch.Tensor:
        img_min = img.detach().min()
        img_max = img.detach().max()
        if img_min.item() >= 0.0 and img_max.item() <= 1.0:
            return self.normalize(img)
        return img

    def forward(self, img: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        b, _, h, w, device, img_size = (
            *img.shape,
            img.device,
            self.image_size,
        )
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if h != img_size[0] or w != img_size[1]:
            raise ValueError(f"height and width of image must be {img_size}")
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self._normalize_if_needed(img)
        return self.p_losses(img, t, *args, **kwargs)

    @torch.inference_mode()
    def sample(self, batch_size: int = 16, return_all_timesteps: bool = False):
        image_size, channels = self.image_size, self.channels
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(
            (batch_size, channels, image_size[0], image_size[1]), return_all_timesteps=return_all_timesteps
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        offset_noise_strength: float | None = None,
    ) -> torch.Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        model_out = self.model(x, t, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if self.loss_type == "l1":
            loss = F.l1_loss(model_out, target, reduction="none")
        else:
            loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()


class GaussianDiffusionGenerator(FIDValidationMixin, SampleVisualizationMixin, pl.LightningModule):
    """Lightning wrapper for Gaussian diffusion with denoising_diffusion_pytorch UNet."""

    def __init__(
        self,
        lr: float = 1e-4,
        sample_size: int | tuple[int, int] = 32,
        num_timesteps: int = 1000,
        sampling_timesteps: int | None = None,
        save_dir: str = "samples",
        sampling_every_n_steps: int = 0,
        sampling_num_images: int = 16,
        sampling_grid_nrow: int = 4,
        fid_num_samples: int = 256,
        fid_low_vram_mode: bool = False,
        fid_clear_cuda_cache: bool = True,
        use_cpu_for_fid: bool = False,
        feature_maps: int = 64,
        dim_mults: Sequence[int] | None = None,
        objective: Literal["pred_noise", "pred_x0", "pred_v"] = "pred_v",
        beta_schedule: Literal["linear", "cosine", "sigmoid"] = "sigmoid",
        schedule_fn_kwargs: dict[str, Any] | None = None,
        ddim_sampling_eta: float = 0.0,
        auto_normalize: bool = True,
        offset_noise_strength: float = 0.0,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5.0,
        self_condition: bool = False,
        loss_type: Literal["l1", "l2"] = "l2",
        use_medicraft_settings: bool = True,
        **_: Any,
    ) -> None:
        pl.LightningModule.__init__(self)
        FIDValidationMixin.__init__(
            self,
            fid_num_samples=fid_num_samples,
            low_vram_mode=fid_low_vram_mode,
            clear_cuda_cache=fid_clear_cuda_cache,
            use_cpu_for_fid=use_cpu_for_fid,
        )
        self.sample_size = normalize_hw(sample_size, field_name="model.sample_size")
        self.lr = lr
        self.save_dir = Path(save_dir)
        self.sampling_every_n_steps = sampling_every_n_steps
        self.sampling_num_images = sampling_num_images
        self.sampling_grid_nrow = sampling_grid_nrow
        self.feature_maps = feature_maps
        self.dim_mults = tuple(dim_mults or (1, 2, 4, 8))
        self.loss_type = loss_type

        if use_medicraft_settings:
            unet = Unet(
                dim=64,
                dim_mults=(1, 2, 4, 8),
                channels=1,
            )
            self.diffusion = GaussianDiffusion(
                unet,
                image_size=self.sample_size,
                timesteps=num_timesteps,
                sampling_timesteps=sampling_timesteps,  # Enable DDIM for faster sampling
            )
        else:
            unet = Unet(
                dim=self.feature_maps,
                dim_mults=self.dim_mults,
                channels=1,
                self_condition=self_condition,
            )
            self.diffusion = GaussianDiffusion(
                unet,
                image_size=self.sample_size,
                timesteps=num_timesteps,
                sampling_timesteps=sampling_timesteps,
                objective=objective,
                beta_schedule=beta_schedule,
                schedule_fn_kwargs=schedule_fn_kwargs or {},
                ddim_sampling_eta=ddim_sampling_eta,
                auto_normalize=auto_normalize,
                offset_noise_strength=offset_noise_strength,
                min_snr_loss_weight=min_snr_loss_weight,
                min_snr_gamma=min_snr_gamma,
                loss_type=loss_type,
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.diffusion(images)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        images, _ = batch
        loss = self(images)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99))

    @torch.inference_mode()
    def sample(self, num_samples: int = 16) -> torch.Tensor:
        return self.diffusion.sample(batch_size=num_samples)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.sampling_every_n_steps <= 0:
            return
        if self.global_step == 0 or self.global_step % self.sampling_every_n_steps != 0:
            return
        self._log_and_optionally_track_samples(tag="generated_samples", caption=f"step {self.global_step}")

    def on_train_end(self) -> None:
        samples = self.sample(num_samples=self.sampling_num_images)
        grid, grid_path = self._save_samples_grid(samples, step=None, prefix="gaussian_diffusion_samples")
        self.print(f"Saved samples to {grid_path}")
        logger = self.logger
        if isinstance(logger, WandbLogger):
            logger.log_image(
                key="gaussian_diffusion_final_samples",
                images=[grid],
                caption=["training_end"],
            )

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if self.sampling_every_n_steps > 0:
            self._log_and_optionally_track_samples(
                tag="validation_samples",
                caption=f"validation-step {self.global_step}",
            )

    def _prepare_samples_grid_view(self) -> tuple[torch.Tensor, Path]:
        samples = self.sample(num_samples=self.sampling_num_images)
        return self._save_samples_grid(samples, step=self.global_step, prefix="gaussian_diffusion_samples")

    def _log_and_optionally_track_samples(self, *, tag: str, caption: str) -> None:
        if getattr(self, "global_rank", 0) != 0:
            return
        grid, grid_path = self._prepare_samples_grid_view()
        self.print(f"Saved intermediate samples to {grid_path}")
        logger = self.logger
        if isinstance(logger, WandbLogger):
            logger.log_image(
                key=tag,
                images=[grid],
                caption=[caption],
            )

    @torch.inference_mode()
    def generate_samples_for_fid(self, num_samples: int) -> torch.Tensor:
        self.print(f"[fid] generating {num_samples} samples for FID")
        samples = self.sample(num_samples=num_samples)
        samples = samples.clamp(0, 1)
        self.print("[fid] generated samples for FID")
        return samples
