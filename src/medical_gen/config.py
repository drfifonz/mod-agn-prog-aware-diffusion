"""Configuration utilities for DDPM training."""

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .utils import normalize_hw

PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, dotted))
        else:
            flattened[dotted] = value
    return flattened


def _resolve_placeholders(obj: Any, values: Dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {key: _resolve_placeholders(value, values) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_resolve_placeholders(item, values) for item in obj]
    if isinstance(obj, str):
        full_match = PLACEHOLDER_PATTERN.fullmatch(obj)
        if full_match:
            key = full_match.group(1)
            if key not in values:
                raise ValueError(f"Unknown placeholder '{key}' in configuration.")
            return values[key]

        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise ValueError(f"Unknown placeholder '{key}' in configuration.")
            return str(values[key])

        return PLACEHOLDER_PATTERN.sub(_replace, obj)
    return obj


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _env_overrides(
    prefixes: tuple[str, ...] | None = ("DDPM_", ""),
    delimiter: str = "__",
) -> Dict[str, Any]:
    """Collect environment variables with the given prefixes into a nested dict."""
    overrides: Dict[str, Any] = {}
    for key, value in os.environ.items():
        matched_path: str | None = None
        if prefixes:
            for prefix in prefixes:
                if prefix and key.startswith(prefix):
                    matched_path = key[len(prefix) :]
                    break
                if not prefix and delimiter in key:
                    matched_path = key
                    break
        elif delimiter in key:
            matched_path = key
        if not matched_path:
            continue
        path = matched_path.lower().split(delimiter)
        # Prefer explicit config values for WandB project/entity over env defaults
        if len(path) >= 2 and path[0] == "wandb" and path[1] in {"project", "entity"}:
            continue
        cursor = overrides
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value
    return overrides


class DatasetName(str, Enum):
    MANIFEST = "manifest"
    MNIST = "mnist"
    OCT = "oct"
    CHEST_XRAY = "chest_xray"
    OPHTHAL_ANONYM = "ophthal_anonym"


class DatasetSelection(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    name: DatasetName = DatasetName.MNIST
    params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "data_dir": "data",
            "image_size": 32,
        }
    )


class DataConfig(BaseModel):
    dataset: DatasetSelection = Field(default_factory=DatasetSelection)
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True


class ModelName(str, Enum):
    DDPM = "ddpm_generator"
    DDPM_V2 = "ddpm_generator_v2"
    GAN = "gan_generator"
    GAUSSIAN_DIFFUSION = "gaussian_diffusion_generator"
    SMOTE = "smote_generator"


class ModelConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    name: ModelName = ModelName.DDPM
    lr: float = 1e-4
    sample_size: tuple[int, int] = (32, 32)
    num_timesteps: int = 1000
    save_dir: str = "samples"
    latent_dim: int = 128
    feature_maps: int = 64
    dim_mults: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    beta_schedule: str = "linear"
    beta_start: float | None = None
    beta_end: float | None = None
    ema_decay: float = 0.995
    ema_use_v2: bool = False
    # DDPM V2-specific options
    sampling_timesteps: int | None = None  # None = use num_timesteps (DDPM), < num_timesteps = DDIM
    objective: str = "pred_noise"  # pred_noise, pred_x0, pred_v
    ddim_sampling_eta: float = 0.0  # 0 = deterministic DDIM, 1 = DDPM
    offset_noise_strength: float = 0.0  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
    min_snr_loss_weight: bool = False  # https://arxiv.org/abs/2303.09556
    min_snr_gamma: float = 5.0
    self_condition: bool = False  # Self-conditioning (slows training ~25%, but lowers FID)
    auto_normalize: bool = True  # If true, normalize [0,1] -> [-1,1] for Gaussian diffusion
    loss_type: str = "l2"  # Gaussian diffusion loss: l1 or l2
    use_medicraft_settings: bool = False  # Use Medicraft defaults for Gaussian diffusion
    smote_k_neighbors: int = 5
    # GAN-specific options
    lr_d: float | None = None  # Discriminator LR for TTUR (default: 4x generator lr)
    use_improved_arch: bool = True  # Use improved architecture with self-attention for 128x256
    use_hinge_loss: bool = True  # Hinge loss instead of BCE for stability
    label_smoothing: float = 0.1  # Label smoothing for discriminator
    n_critic: int = 1  # Discriminator updates per generator update

    @model_validator(mode="before")
    @classmethod
    def _normalize_sample_size(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict) and "sample_size" in data:
            data = dict(data)
            data["sample_size"] = normalize_hw(data["sample_size"], field_name="model.sample_size")
        return data


class TrainerConfig(BaseModel):
    max_epochs: int = 1
    max_steps: int | None = None
    limit_train_batches: float = 1.0
    precision: str | int = "16-mixed"
    accelerator: str = "auto"
    devices: str | int | list[int] = "auto"
    log_every_n_steps: int = 10
    num_sanity_val_steps: int = 2
    gradient_accumulate_every: int = 1


class PathsConfig(BaseModel):
    output_root: str = "data/outputs"
    input_root: str = "data"
    experiment_name: str = "generator"
    init_checkpoint: str | None = None


class ValidationConfig(BaseModel):
    enabled: bool = True
    every_n_steps: int | None = 200
    every_n_epochs: int | None = 1
    fid_enabled: bool = True
    fid_num_samples: int = 256
    low_vram_mode: bool = False
    clear_cuda_cache: bool = True
    use_cpu_for_fid: bool = False

    @model_validator(mode="after")
    def _ensure_frequency(self) -> "ValidationConfig":
        if self.enabled and self.every_n_steps is None and self.every_n_epochs is None:
            raise ValueError("Validation is enabled but neither 'every_n_steps' nor 'every_n_epochs' was provided.")
        return self


class SamplingConfig(BaseModel):
    every_n_steps: int = 500
    num_samples: int = 16
    num_inference_steps: int = 50
    grid_nrow: int = 4


class WandBConfig(BaseModel):
    enabled: bool = False
    project: str = "generative-augmentation"
    entity: str | None = None
    run_name: str | None = None
    dir: str = "."
    mode: str = "online"
    log_model: bool = False
    upload_model_artifact: bool = False
    sweep_id: str | None = None


class ExperimentConfig(BaseModel):
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)

    @model_validator(mode="before")
    @classmethod
    def _resolve_template_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return data
        flattened = _flatten_dict(data)
        return _resolve_placeholders(data, flattened)

    @model_validator(mode="after")
    def _ensure_dataset_paths(self) -> "ExperimentConfig":
        params = dict(self.data.dataset.params)
        input_root = Path(self.paths.input_root).expanduser()
        params.setdefault("data_dir", str(input_root))
        images_subdir = params.get("images_subdir")
        if images_subdir and not params.get("images_dir"):
            combined = input_root / images_subdir
            if not combined.is_absolute():
                combined = combined.expanduser().absolute()
            params["images_dir"] = str(combined)
        self.data.dataset.params = params
        init_ckpt = self.paths.init_checkpoint
        if init_ckpt:
            expanded = os.path.expandvars(str(init_ckpt))
            self.paths.init_checkpoint = str(Path(expanded).expanduser())
        return self

    @model_validator(mode="after")
    def _propagate_wandb_defaults(self) -> "ExperimentConfig":
        if self.wandb.enabled and not self.wandb.project:
            raise ValueError("WandB logging is enabled but 'wandb.project' is not set.")
        return self

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        env_prefixes: tuple[str, ...] | None = ("DDPM_", ""),
        overrides: Dict[str, Any] | None = None,
    ) -> "ExperimentConfig":
        """Load config from YAML + env vars, then apply optional overrides."""
        load_dotenv()
        yaml_data: Dict[str, Any] = {}
        if path:
            yaml_path = Path(path)
            if not yaml_path.exists():
                raise FileNotFoundError(f"Config file '{yaml_path}' was not found.")
            yaml_data = yaml.safe_load(yaml_path.read_text()) or {}
        merged = deep_update({}, cls().model_dump())
        merged = deep_update(merged, yaml_data)
        merged = deep_update(merged, _env_overrides(prefixes=env_prefixes))
        if overrides:
            merged = deep_update(merged, overrides)
        return cls.model_validate(merged)
