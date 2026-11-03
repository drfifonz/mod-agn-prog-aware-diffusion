"""DDPM-based generator module."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms

from ..utils import normalize_hw, normalize_to_neg_one_to_one
from ..utils.ema import EMA, EMASchedule
from ..utils.ema_v2 import EMAV2
from .mixins import FIDValidationMixin, SampleVisualizationMixin


class DDPMGenerator(FIDValidationMixin, SampleVisualizationMixin, pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        sample_size: int = 32,
        num_timesteps: int = 1000,
        save_dir: str = "samples",
        sampling_every_n_steps: int = 0,
        sampling_num_images: int = 16,
        sampling_num_inference_steps: int = 50,
        sampling_grid_nrow: int = 4,
        fid_num_samples: int = 256,
        fid_low_vram_mode: bool = False,
        fid_clear_cuda_cache: bool = True,
        use_cpu_for_fid: bool = False,
        feature_maps: int = 64,
        dim_mults: Sequence[int] | None = None,
        beta_schedule: str = "linear",
        beta_start: float | None = None,
        beta_end: float | None = None,
        data_mean: Sequence[float] | float = 0.5,
        data_std: Sequence[float] | float = 0.5,
        ema_decay: float = 0.995,
        ema_update_every: int = 1,
        ema_warmup_steps: int = 0,
        ema_power: float = 1.0,
        ema_use_cpu: bool = False,
        ema_use_v2: bool = False,
        use_ema_for_validation: bool = True,
        use_ema_for_sampling: bool = True,
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
        self.sampling_num_inference_steps = sampling_num_inference_steps
        self.sampling_grid_nrow = sampling_grid_nrow
        self.feature_maps = feature_maps
        self.dim_mults = tuple(dim_mults or (1, 2, 4, 8))
        self.data_mean = self._ensure_tuple(data_mean, "data_mean")
        self.data_std = self._ensure_tuple(data_std, "data_std")
        if len(self.data_mean) != len(self.data_std):
            raise ValueError("data_mean and data_std must have the same length.")
        if any(std == 0 for std in self.data_std):
            raise ValueError("data_std entries must be non-zero to build an inverse transform.")
        self._inverse_normalize: transforms.Compose | None = None
        self.ema_decay = float(ema_decay)
        if self.ema_decay < 0:
            raise ValueError("ema_decay must be non-negative.")
        self.ema_update_every = int(ema_update_every)
        if self.ema_update_every < 1:
            raise ValueError("ema_update_every must be >= 1.")
        self.ema_warmup_steps = int(ema_warmup_steps)
        if self.ema_warmup_steps < 0:
            raise ValueError("ema_warmup_steps must be >= 0.")
        self.ema_power = float(ema_power)
        if self.ema_power <= 0:
            raise ValueError("ema_power must be > 0.")
        self.ema_on_cpu = bool(ema_use_cpu)
        self.ema_use_v2 = bool(ema_use_v2)
        self.use_ema_for_validation = bool(use_ema_for_validation)
        self.use_ema_for_sampling = bool(use_ema_for_sampling)
        self.ema: EMA | None = None
        self.ema_v2: EMAV2 | None = None
        self._ema_schedule = (
            EMASchedule(base_decay=self.ema_decay, warmup_steps=self.ema_warmup_steps, power=self.ema_power)
            if self.ema_warmup_steps > 0 and self.ema_decay > 0
            else None
        )
        self._ema_step: int = 0
        self._ema_in_use: bool = False
        self._ema_state_to_load: dict[str, torch.Tensor] | None = None
        if not self.dim_mults:
            raise ValueError("dim_mults must contain at least one multiplier.")
        self.block_out_channels = tuple(self.feature_maps * m for m in self.dim_mults)
        down_block_types = self._build_down_block_types(len(self.block_out_channels))
        up_block_types = self._build_up_block_types(len(self.block_out_channels))
        self.model = UNet2DModel(
            sample_size=self.sample_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=8,
            block_out_channels=self.block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
        self.ema_target = self.model
        scheduler_kwargs: dict[str, Any] = {
            "num_train_timesteps": num_timesteps,
            "beta_schedule": beta_schedule,
        }
        schedule_start = beta_start
        schedule_end = beta_end
        if beta_schedule == "sigmoid":  # prefer stable defaults for sigmoid schedules
            schedule_start = schedule_start if schedule_start is not None else 0.0001
            schedule_end = schedule_end if schedule_end is not None else 0.02
        if schedule_start is not None:
            scheduler_kwargs["beta_start"] = schedule_start
        if schedule_end is not None:
            scheduler_kwargs["beta_end"] = schedule_end

        self.noise_scheduler = DDPMScheduler(**scheduler_kwargs)

    def _ema_enabled(self) -> bool:
        return self.ema_decay > 0

    @staticmethod
    def _ensure_tuple(value: Sequence[float] | float, field_name: str) -> tuple[float, ...]:
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{field_name} must be a float or sequence of floats, not a string.")
        if isinstance(value, Sequence):
            return tuple(float(v) for v in value)
        return (float(value),)

    def _get_inverse_normalize_transform(self) -> transforms.Compose:
        if self._inverse_normalize is None:
            zeros = [0.0] * len(self.data_mean)
            inv_std = [1.0 / std for std in self.data_std]
            negative_mean = [-mean for mean in self.data_mean]
            ones = [1.0] * len(self.data_mean)
            self._inverse_normalize = transforms.Compose(
                [
                    transforms.Normalize(mean=zeros, std=inv_std),
                    transforms.Normalize(mean=negative_mean, std=ones),
                ]
            )
        return self._inverse_normalize

    def _ema_available(self) -> bool:
        if not self._ema_enabled():
            return False
        return (self.ema_v2 is not None) if self.ema_use_v2 else (self.ema is not None)

    def _maybe_init_ema(self) -> None:
        if not self._ema_enabled():
            return
        if (self.ema_v2 is not None and self.ema_use_v2) or (self.ema is not None and not self.ema_use_v2):
            return
        device = torch.device("cpu") if self.ema_on_cpu else self.device
        if self.ema_use_v2:
            self.ema_v2 = EMAV2(
                self.ema_target,
                beta=self.ema_decay,
                update_after_step=self.ema_warmup_steps,
                update_every=self.ema_update_every,
                inv_gamma=max(1.0, float(self.ema_warmup_steps) if self.ema_warmup_steps > 0 else 1.0),
                power=self.ema_power,
                ema_device=device,
                allow_different_devices=self.ema_on_cpu,
            )
            if self._ema_state_to_load is not None:
                self.ema_v2.load_state_dict(self._ema_state_to_load)
                self._ema_state_to_load = None
        else:
            self.ema = EMA(self.ema_target, decay=self.ema_decay, device=device)
            if self._ema_state_to_load is not None:
                self.ema.load_state_dict(self._ema_state_to_load)
                self._ema_state_to_load = None

    def _ema_decay_value(self) -> float:
        if self.ema_use_v2:
            return self.ema_decay
        if self._ema_schedule is None:
            return self.ema_decay
        return self._ema_schedule(self._ema_step)

    def _update_ema(self) -> None:
        self._maybe_init_ema()
        if not self._ema_available():
            return
        self._ema_step += 1
        if self._ema_step % self.ema_update_every != 0:
            return
        decay = self._ema_decay_value()
        if self.ema_use_v2:
            self.ema_v2.update(self.ema_target, decay=decay)  # type: ignore[union-attr]
        else:
            self.ema.update(self.ema_target, decay=decay)  # type: ignore[union-attr]

    def _activate_ema_weights(self) -> bool:
        self._maybe_init_ema()
        if not self._ema_available() or self._ema_in_use:
            return False
        if self.ema_use_v2:
            self.ema_v2.store(self.ema_target)  # type: ignore[union-attr]
            self.ema_v2.copy_to(self.ema_target)  # type: ignore[union-attr]
        else:
            self.ema.store(self.ema_target)  # type: ignore[union-attr]
            self.ema.copy_to(self.ema_target)  # type: ignore[union-attr]
        self._ema_in_use = True
        return True

    def _restore_ema_weights(self) -> None:
        if not self._ema_in_use or not self._ema_available():
            return
        if self.ema_use_v2:
            self.ema_v2.restore(self.ema_target)  # type: ignore[union-attr]
        else:
            self.ema.restore(self.ema_target)  # type: ignore[union-attr]
        self._ema_in_use = False

    @contextmanager
    def _maybe_use_ema_weights(self, enabled: bool):
        if not enabled or not self._ema_available():
            yield False
            return
        if self._ema_in_use:
            yield True
            return
        self._activate_ema_weights()
        try:
            yield True
        finally:
            self._restore_ema_weights()

    def on_fit_start(self) -> None:
        self._maybe_init_ema()
        super().on_fit_start()

    @staticmethod
    def _build_down_block_types(num_blocks: int) -> tuple[str, ...]:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive for UNet construction.")
        down_blocks: list[str] = []
        for idx in range(num_blocks):
            down_blocks.append("DownBlock2D" if idx == 0 else "AttnDownBlock2D")
        return tuple(down_blocks)

    @staticmethod
    def _build_up_block_types(num_blocks: int) -> tuple[str, ...]:
        up_blocks: list[str] = []
        for idx in range(num_blocks):
            up_blocks.append("UpBlock2D" if idx == num_blocks - 1 else "AttnUpBlock2D")
        return tuple(up_blocks)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        noise_pred = self.model(x, timesteps, return_dict=False)[0]
        return noise_pred

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        images, _ = batch
        loss = self._diffusion_loss(images)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def _diffusion_loss(self, images: torch.Tensor) -> torch.Tensor:
        img_min = images.detach().min()
        img_max = images.detach().max()
        if img_min.item() >= 0.0 and img_max.item() <= 1.0:
            images = normalize_to_neg_one_to_one(images)
        noise = torch.randn_like(images)
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
        noise_pred = self(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99))

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        num_inference_steps: int = 50,
        use_ema: bool | None = None,
    ) -> torch.Tensor:
        use_ema = self.use_ema_for_sampling if use_ema is None else use_ema
        if use_ema:
            self._maybe_init_ema()
        use_ema = use_ema and self._ema_available()
        was_training = self.model.training
        self.model.eval()
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        height, width = self.sample_size
        images = torch.randn(
            (num_samples, 1, height, width),
            device=self.device,
        )
        logger = self.logger
        total_steps = len(self.noise_scheduler.timesteps)
        device_type = getattr(self.device, "type", "cpu")
        use_autocast = device_type == "cuda"
        autocast_kwargs: dict[str, Any] | None = None
        if use_autocast:
            autocast_kwargs = {"device_type": device_type, "dtype": torch.float16}
        wandb_progress: list[str] = []
        with self._maybe_use_ema_weights(use_ema):
            for idx, t in enumerate(self.noise_scheduler.timesteps, start=1):
                if autocast_kwargs is not None:
                    with torch.autocast(**autocast_kwargs):
                        noise_pred = self.model(images, t, return_dict=False)[0]
                else:
                    noise_pred = self.model(images, t, return_dict=False)[0]
                noise_pred = noise_pred.float()
                step_output = self.noise_scheduler.step(noise_pred, t, images)
                images = step_output.prev_sample
                if idx % max(1, total_steps // 10) == 0:
                    step_info = f"sampling step {idx}/{total_steps}"
                    if logger is None:
                        self.print(step_info)
                    elif isinstance(logger, WandbLogger):
                        wandb_progress.append(step_info)
                    else:
                        self.print(step_info)
            if wandb_progress and isinstance(logger, WandbLogger):
                logger.log_text(
                    "sampling_progress",
                    columns=["status"],
                    data=[[entry] for entry in wandb_progress],
                )
        inverse_normalize = self._get_inverse_normalize_transform()
        images = inverse_normalize(images)
        if was_training:
            self.model.train()
        return images.cpu()

    def on_train_end(self) -> None:
        samples = self.sample(
            num_samples=self.sampling_num_images,
            num_inference_steps=self.sampling_num_inference_steps,
        )
        grid, grid_path = self._save_samples_grid(
            samples,
            step=None,
            prefix="ddpm_samples",
        )
        self.print(f"Saved samples to {grid_path}")
        logger = self.logger
        if isinstance(logger, WandbLogger):
            logger.log_image(
                key="ddpm_final_samples",
                images=[grid],
                caption=["training_end"],
            )

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self._update_ema()
        if self.sampling_every_n_steps <= 0:
            return
        if self.global_step == 0 or self.global_step % self.sampling_every_n_steps != 0:
            return
        self._log_and_optionally_track_samples(tag="generated_samples", caption=f"step {self.global_step}")

    def on_validation_epoch_start(self) -> None:
        if self._fid_low_vram_mode:
            super().on_validation_epoch_start()
            if self.use_ema_for_validation:
                self._activate_ema_weights()
        else:
            if self.use_ema_for_validation:
                self._activate_ema_weights()
            super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if self.sampling_every_n_steps > 0:
            self._log_and_optionally_track_samples(
                tag="validation_samples",
                caption=f"validation-step {self.global_step}",
            )
        self._restore_ema_weights()

    def _prepare_samples_grid_view(self) -> tuple[torch.Tensor, Path]:
        samples = self.sample(
            num_samples=self.sampling_num_images,
            num_inference_steps=self.sampling_num_inference_steps,
        )
        return self._save_samples_grid(
            samples,
            step=self.global_step,
            prefix="ddpm_samples",
        )

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

    def generate_samples_for_fid(self, num_samples: int) -> torch.Tensor:
        samples = self.sample(
            num_samples=num_samples,
            num_inference_steps=self.sampling_num_inference_steps,
            use_ema=True,
        )
        return samples

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self._ema_available():
            ema_state = self.ema_v2.state_dict() if self.ema_use_v2 else self.ema.state_dict()  # type: ignore[union-attr]
            checkpoint["ema_state_dict"] = ema_state
            checkpoint["ema_step"] = self._ema_step
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint.get("ema_state_dict")
        if state is not None:
            self._ema_state_to_load = state
        if "ema_step" in checkpoint:
            self._ema_step = int(checkpoint["ema_step"])
        super().on_load_checkpoint(checkpoint)
