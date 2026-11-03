"""DDPM-based generator module V2 - simplified diffusion math inspired by denoising_diffusion_pytorch."""

from __future__ import annotations

import math
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from random import random
from typing import Any, Literal, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from tqdm import tqdm

from ..utils import normalize_hw
from ..utils.ema import EMA, EMASchedule
from ..utils.ema_v2 import EMAV2
from .mixins import FIDValidationMixin, SampleVisualizationMixin

# ============================================================================
# Helper functions (from denoising_diffusion_pytorch)
# ============================================================================


def reduce_mean(x: torch.Tensor, dims_to_reduce: str = "b ... -> b") -> torch.Tensor:
    """Reduce tensor by computing mean over all dimensions except batch."""
    # Simple implementation: mean over all dimensions except batch (dim 0)
    if x.dim() == 1:
        return x
    return x.flatten(1).mean(dim=1)


def exists(x: Any) -> bool:
    return x is not None


def default(val: Any, d: Any) -> Any:
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t: Any, *args: Any, **kwargs: Any) -> Any:
    return t


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    """Extract values from a at indices t and reshape for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# ============================================================================
# Normalization functions
# ============================================================================


def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    return img * 2 - 1


def unnormalize_to_zero_to_one(t: torch.Tensor) -> torch.Tensor:
    return (t + 1) * 0.5


# ============================================================================
# Beta schedules (from denoising_diffusion_pytorch)
# ============================================================================


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Linear schedule, proposed in original DDPM paper."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(
    timesteps: int,
    start: float = -3,
    end: float = 3,
    tau: float = 1,
    clamp_min: float = 1e-5,
) -> torch.Tensor:
    """Sigmoid schedule proposed in https://arxiv.org/abs/2212.11972 - Figure 8.

    Better for images > 64x64, when used during training.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# ============================================================================
# DDPMGeneratorV2 - Simplified diffusion with PyTorch Lightning
# ============================================================================


class DDPMGeneratorV2(FIDValidationMixin, SampleVisualizationMixin, pl.LightningModule):
    """DDPM Generator V2 with simplified diffusion math from denoising_diffusion_pytorch.

    Key differences from V1:
    - Inline diffusion math (no DDPMScheduler dependency)
    - Support for pred_noise, pred_x0, pred_v objectives
    - Support for linear, cosine, sigmoid beta schedules
    - Optional DDIM sampling with configurable eta
    - Min-SNR loss weighting
    - Offset noise for improved generation
    - Self-conditioning support
    """

    def __init__(
        self,
        lr: float = 1e-4,
        sample_size: int = 32,
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
        # Diffusion config
        objective: Literal["pred_noise", "pred_x0", "pred_v"] = "pred_noise",
        beta_schedule: Literal["linear", "cosine", "sigmoid"] = "linear",
        schedule_fn_kwargs: dict[str, Any] | None = None,
        ddim_sampling_eta: float = 0.0,
        auto_normalize: bool = True,
        offset_noise_strength: float = 0.0,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5.0,
        self_condition: bool = False,
        # Data normalization
        data_mean: Sequence[float] | float = 0.5,
        data_std: Sequence[float] | float = 0.5,
        # EMA config
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

        # Basic config
        self.sample_size = normalize_hw(sample_size, field_name="model.sample_size")
        self.lr = lr
        self.save_dir = Path(save_dir)
        self.sampling_every_n_steps = sampling_every_n_steps
        self.sampling_num_images = sampling_num_images
        self.sampling_grid_nrow = sampling_grid_nrow
        self.feature_maps = feature_maps
        self.dim_mults = tuple(dim_mults or (1, 2, 4, 8))
        self.channels = 1  # Grayscale for medical images

        # Data normalization
        self.data_mean = self._ensure_tuple(data_mean, "data_mean")
        self.data_std = self._ensure_tuple(data_std, "data_std")
        if len(self.data_mean) != len(self.data_std):
            raise ValueError("data_mean and data_std must have the same length.")
        if any(std == 0 for std in self.data_std):
            raise ValueError("data_std entries must be non-zero.")
        self._inverse_normalize: transforms.Compose | None = None

        # Diffusion config
        self.objective = objective
        self.self_condition = self_condition
        self.offset_noise_strength = offset_noise_strength

        if objective not in {"pred_noise", "pred_x0", "pred_v"}:
            raise ValueError(f"objective must be 'pred_noise', 'pred_x0', or 'pred_v', got {objective}")

        # Build beta schedule
        schedule_fn_kwargs = schedule_fn_kwargs or {}
        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        betas = beta_schedule_fn(num_timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(num_timesteps)

        # Sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, num_timesteps)
        assert self.sampling_timesteps <= num_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # Register buffers (float64 -> float32)
        def register_buffer(name: str, val: torch.Tensor) -> None:
            self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # SNR loss weighting
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif objective == "pred_x0":
            register_buffer("loss_weight", maybe_clipped_snr)
        elif objective == "pred_v":
            register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

        # Auto-normalization
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        # Build UNet model
        if not self.dim_mults:
            raise ValueError("dim_mults must contain at least one multiplier.")
        self.block_out_channels = tuple(self.feature_maps * m for m in self.dim_mults)
        down_block_types = self._build_down_block_types(len(self.block_out_channels))
        up_block_types = self._build_up_block_types(len(self.block_out_channels))

        # Adjust input channels for self-conditioning
        in_channels = self.channels * (2 if self.self_condition else 1)

        self.model = UNet2DModel(
            sample_size=self.sample_size,
            in_channels=in_channels,
            out_channels=self.channels,
            layers_per_block=2,
            block_out_channels=self.block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
        self.ema_target = self.model

        # EMA config
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
            EMASchedule(
                base_decay=self.ema_decay,
                warmup_steps=self.ema_warmup_steps,
                power=self.ema_power,
            )
            if self.ema_warmup_steps > 0 and self.ema_decay > 0
            else None
        )
        self._ema_step: int = 0
        self._ema_in_use: bool = False
        self._ema_state_to_load: dict[str, torch.Tensor] | None = None

    # ========================================================================
    # Helper methods
    # ========================================================================

    @staticmethod
    def _ensure_tuple(value: Sequence[float] | float, field_name: str) -> tuple[float, ...]:
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{field_name} must be a float or sequence of floats.")
        if isinstance(value, Sequence):
            return tuple(float(v) for v in value)
        return (float(value),)

    @staticmethod
    def _build_down_block_types(num_blocks: int) -> tuple[str, ...]:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")
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

    # ========================================================================
    # EMA methods (same as V1)
    # ========================================================================

    def _ema_enabled(self) -> bool:
        return self.ema_decay > 0

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

    # ========================================================================
    # Diffusion math (from denoising_diffusion_pytorch)
    # ========================================================================

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Predict noise from x_t and x_0."""
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict v (velocity) from x_0 and noise."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and v."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Sample from q(x_t | x_0) - the forward diffusion process."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
        clip_x_start: bool = False,
        rederive_pred_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get model predictions (pred_noise, pred_x_start)."""
        # Prepare input for self-conditioning
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            model_input = torch.cat((x_self_cond, x), dim=1)
        else:
            model_input = x

        model_output = self.model(model_input, t, return_dict=False)[0]
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mean and variance for p(x_{t-1} | x_t)."""
        pred_noise, x_start = self.model_predictions(x, t, x_self_cond)

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # ========================================================================
    # Sampling methods
    # ========================================================================

    @torch.inference_mode()
    def p_sample(
        self, x: torch.Tensor, t: int, x_self_cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single DDPM sampling step."""
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape: tuple[int, ...], return_all_timesteps: bool = False) -> torch.Tensor:
        """Full DDPM sampling loop."""
        device = self.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)), total=self.num_timesteps, desc="sample-ddpm", leave=False
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape: tuple[int, ...], return_all_timesteps: bool = False) -> torch.Tensor:
        """DDIM sampling loop."""
        batch_size, device = shape[0], self.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, total=len(time_pairs), desc="sample-ddim", leave=False):
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start = self.model_predictions(
                img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
            )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(
        self,
        num_samples: int = 16,
        use_ema: bool | None = None,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        """Generate samples."""
        use_ema = self.use_ema_for_sampling if use_ema is None else use_ema
        if use_ema:
            self._maybe_init_ema()
        use_ema = use_ema and self._ema_available()

        was_training = self.model.training
        self.model.eval()

        height, width = self.sample_size
        shape = (num_samples, self.channels, height, width)

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        self.print(
            f"[sample] start num={num_samples} mode={'ddim' if self.is_ddim_sampling else 'ddpm'} "
            f"ema={use_ema} device={self.device}"
        )

        with self._maybe_use_ema_weights(use_ema):
            device_type = getattr(self.device, "type", "cpu")
            autocast_dtype = torch.bfloat16 if device_type == "cuda" else None
            # Autocast wraps the whole sampling loop (DDPM or DDIM) to speed inference.
            if device_type == "cuda" and autocast_dtype is not None:
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    samples = sample_fn(shape, return_all_timesteps=return_all_timesteps)
            else:
                samples = sample_fn(shape, return_all_timesteps=return_all_timesteps)

        # Apply inverse normalization
        inverse_normalize = self._get_inverse_normalize_transform()
        samples = inverse_normalize(samples)

        if was_training:
            self.model.train()

        self.print("[sample] done")

        return samples.cpu()

    # ========================================================================
    # Training methods
    # ========================================================================

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass - predict based on objective."""
        if self.self_condition:
            # For forward pass without self-conditioning
            x_input = torch.cat((torch.zeros_like(x), x), dim=1)
        else:
            x_input = x
        return self.model(x_input, timesteps, return_dict=False)[0]

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute training loss."""
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # Offset noise
        if self.offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise = noise + self.offset_noise_strength * offset_noise.unsqueeze(-1).unsqueeze(-1)

        # Noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Self-conditioning: 50% of the time, predict x_start and condition with it
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                _, x_self_cond = self.model_predictions(x, t)
                x_self_cond = x_self_cond.detach()

        # Prepare model input
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            model_input = torch.cat((x_self_cond, x), dim=1)
        else:
            model_input = x

        # Predict
        model_out = self.model(model_input, t, return_dict=False)[0]

        # Compute target based on objective
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # Compute loss with SNR weighting
        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce_mean(loss)  # b c h w -> b (mean over c, h, w)
        loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        images, _ = batch

        # Normalize to [-1, 1]
        images = self.normalize(images)

        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (images.shape[0],), device=self.device).long()

        loss = self.p_losses(images, t)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.99))

    # ========================================================================
    # Lightning hooks
    # ========================================================================

    def on_fit_start(self) -> None:
        self._maybe_init_ema()
        super().on_fit_start()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self._update_ema()
        if self.sampling_every_n_steps <= 0:
            return
        if self.global_step == 0 or self.global_step % self.sampling_every_n_steps != 0:
            return
        self._log_and_optionally_track_samples(tag="generated_samples", caption=f"step {self.global_step}")

    def on_train_end(self) -> None:
        samples = self.sample(num_samples=self.sampling_num_images)
        grid, grid_path = self._save_samples_grid(
            samples,
            step=None,
            prefix="ddpm_v2_samples",
        )
        self.print(f"Saved samples to {grid_path}")
        logger = self.logger
        if isinstance(logger, WandbLogger):
            logger.log_image(
                key="ddpm_v2_final_samples",
                images=[grid],
                caption=["training_end"],
            )

    def on_validation_epoch_start(self) -> None:
        self.print(
            f"[val] epoch start step={self.global_step} use_ema={self.use_ema_for_validation} "
            f"low_vram={getattr(self, '_fid_low_vram_mode', False)}"
        )
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
        self.print(f"[val] epoch end step={self.global_step} fid_done")

    def _prepare_samples_grid_view(self) -> tuple[torch.Tensor, Path]:
        samples = self.sample(num_samples=self.sampling_num_images)
        return self._save_samples_grid(
            samples,
            step=self.global_step,
            prefix="ddpm_v2_samples",
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

    @torch.inference_mode()
    def generate_samples_for_fid(self, num_samples: int) -> torch.Tensor:
        self.print(f"[fid] generating {num_samples} samples for FID")
        samples = self.sample(num_samples=num_samples, use_ema=True)
        self.print("[fid] generated samples for FID")
        return samples

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self._ema_available():
            ema_state = (
                self.ema_v2.state_dict() if self.ema_use_v2 else self.ema.state_dict()
            )  # type: ignore[union-attr]
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
