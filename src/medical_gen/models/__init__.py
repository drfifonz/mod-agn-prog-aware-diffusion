"""Model registry and factory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

import pytorch_lightning as pl

from ..config import ModelConfig, ModelName, SamplingConfig, ValidationConfig
from .ddpm import DDPMGenerator
from .ddpm_v2 import DDPMGeneratorV2
from .gan import GANGenerator
from .gausian_diffusion import GaussianDiffusionGenerator
from .smote import SMOTEGenerator

MODEL_REGISTRY: Dict[ModelName, Type[pl.LightningModule]] = {
    ModelName.DDPM: DDPMGenerator,
    ModelName.DDPM_V2: DDPMGeneratorV2,
    ModelName.GAN: GANGenerator,
    ModelName.GAUSSIAN_DIFFUSION: GaussianDiffusionGenerator,
    ModelName.SMOTE: SMOTEGenerator,
}


def build_generator(
    model_cfg: ModelConfig,
    sampling_cfg: SamplingConfig,
    validation_cfg: ValidationConfig,
    samples_dir: Path,
) -> pl.LightningModule:
    try:
        model_cls = MODEL_REGISTRY[model_cfg.name]
    except KeyError as err:
        available = ", ".join(sorted(name.value for name in MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_cfg.name}'. Available: {available}") from err

    # Common parameters for all models
    common_kwargs = dict(
        lr=model_cfg.lr,
        sample_size=model_cfg.sample_size,
        num_timesteps=model_cfg.num_timesteps,
        save_dir=str(samples_dir),
        sampling_every_n_steps=sampling_cfg.every_n_steps,
        sampling_num_images=sampling_cfg.num_samples,
        sampling_grid_nrow=sampling_cfg.grid_nrow,
        fid_num_samples=validation_cfg.fid_num_samples,
        fid_enabled=validation_cfg.enabled and validation_cfg.fid_enabled,
        fid_low_vram_mode=validation_cfg.low_vram_mode,
        fid_clear_cuda_cache=validation_cfg.clear_cuda_cache,
        use_cpu_for_fid=validation_cfg.use_cpu_for_fid,
        feature_maps=model_cfg.feature_maps,
        dim_mults=model_cfg.dim_mults,
        beta_schedule=model_cfg.beta_schedule,
        ema_decay=model_cfg.ema_decay,
        ema_use_v2=model_cfg.ema_use_v2,
    )

    # Model-specific parameters
    if model_cfg.name == ModelName.DDPM:
        common_kwargs.update(
            sampling_num_inference_steps=sampling_cfg.num_inference_steps,
            beta_start=model_cfg.beta_start,
            beta_end=model_cfg.beta_end,
        )
    elif model_cfg.name == ModelName.DDPM_V2:
        common_kwargs.update(
            sampling_timesteps=getattr(model_cfg, "sampling_timesteps", None),
            objective=getattr(model_cfg, "objective", "pred_noise"),
            ddim_sampling_eta=getattr(model_cfg, "ddim_sampling_eta", 0.0),
            offset_noise_strength=getattr(model_cfg, "offset_noise_strength", 0.0),
            min_snr_loss_weight=getattr(model_cfg, "min_snr_loss_weight", False),
            min_snr_gamma=getattr(model_cfg, "min_snr_gamma", 5.0),
            self_condition=getattr(model_cfg, "self_condition", False),
        )
    elif model_cfg.name == ModelName.GAUSSIAN_DIFFUSION:
        common_kwargs.update(
            sampling_timesteps=getattr(model_cfg, "sampling_timesteps", None),
            objective=getattr(model_cfg, "objective", "pred_v"),
            ddim_sampling_eta=getattr(model_cfg, "ddim_sampling_eta", 0.0),
            offset_noise_strength=getattr(model_cfg, "offset_noise_strength", 0.0),
            min_snr_loss_weight=getattr(model_cfg, "min_snr_loss_weight", False),
            min_snr_gamma=getattr(model_cfg, "min_snr_gamma", 5.0),
            self_condition=getattr(model_cfg, "self_condition", False),
            auto_normalize=getattr(model_cfg, "auto_normalize", True),
            loss_type=getattr(model_cfg, "loss_type", "l2"),
            use_medicraft_settings=getattr(model_cfg, "use_medicraft_settings", True),
        )
    elif model_cfg.name == ModelName.GAN:
        common_kwargs["latent_dim"] = model_cfg.latent_dim
        common_kwargs["lr_d"] = getattr(model_cfg, "lr_d", None)
        common_kwargs["use_improved_arch"] = getattr(model_cfg, "use_improved_arch", True)
        common_kwargs["use_hinge_loss"] = getattr(model_cfg, "use_hinge_loss", True)
        common_kwargs["label_smoothing"] = getattr(model_cfg, "label_smoothing", 0.1)
        common_kwargs["n_critic"] = getattr(model_cfg, "n_critic", 1)
    elif model_cfg.name == ModelName.SMOTE:
        common_kwargs["smote_k_neighbors"] = model_cfg.smote_k_neighbors

    return model_cls(**common_kwargs)


__all__ = ["build_generator", "MODEL_REGISTRY"]
