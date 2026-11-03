"""Config-driven image generation with pluggable models (GAN, DDPM, Gaussian Diffusion)."""

from __future__ import annotations

import argparse
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal

import torch
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torchvision import utils as tv_utils
from tqdm import tqdm

import wandb

from .config import ModelName, deep_update

# Model imports
from .models import MODEL_REGISTRY


def _resolve_path(data: Any, path_parts: List[str]) -> Any:
    """Resolve a dotted path like 'models.0.model_type' through nested dicts/lists."""
    current = data
    for part in path_parts:
        if current is None:
            raise KeyError(f"Cannot traverse into None at path segment '{part}'")
        if isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError) as exc:
                raise KeyError(f"Invalid list index '{part}': {exc}") from exc
        elif isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Key '{part}' not found in dict")
            current = current[part]
        else:
            raise KeyError(f"Cannot traverse into {type(current).__name__} with key '{part}'")
    return current


def _interpolate_value(value: Any, root_data: Dict[str, Any]) -> Any:
    """Interpolate ${...} references in a string value."""
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{([^}]+)\}")

    def _replace(match: re.Match[str]) -> str:
        path = match.group(1)
        path_parts = path.split(".")
        try:
            resolved = _resolve_path(root_data, path_parts)
            return str(resolved)
        except KeyError as exc:
            raise ValueError(f"Cannot resolve interpolation '${{{path}}}': {exc}") from exc

    # Check if full string is a single reference (preserve type)
    full_match = pattern.fullmatch(value)
    if full_match:
        path = full_match.group(1)
        path_parts = path.split(".")
        return _resolve_path(root_data, path_parts)

    # Multiple refs or mixed content - return string
    return pattern.sub(_replace, value)


def _interpolate_config(data: Any, root_data: Dict[str, Any]) -> Any:
    """Recursively interpolate all ${...} references in config data."""
    if isinstance(data, dict):
        return {k: _interpolate_config(v, root_data) for k, v in data.items()}
    elif isinstance(data, list):
        return [_interpolate_config(item, root_data) for item in data]
    elif isinstance(data, str):
        return _interpolate_value(data, root_data)
    return data


class ModelGenerationConfig(BaseModel):
    """Configuration for a single model to generate from."""

    model_config = ConfigDict(use_enum_values=True)

    name: str  # unique identifier for this generation run
    model_type: ModelName  # ddpm_generator, gan_generator, gaussian_diffusion_generator, etc.
    checkpoint_path: str  # path to the checkpoint file
    num_images: int = Field(default=1000, gt=0)  # total number of images to generate
    batch_size: int = Field(default=16, gt=0)  # batch size for generation
    output_dir: str = "generated_images"  # output directory for this model
    # Optional model-specific settings
    num_inference_steps: int | None = None  # for DDPM/diffusion models
    use_ema: bool = True  # whether to use EMA weights if available
    sample_size: tuple[int, int] = (128, 256)  # image dimensions (H, W)
    # Extra model kwargs that will be passed during loading
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_sample_size(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict) and "sample_size" in data:
            data = dict(data)
            raw = data["sample_size"]
            if isinstance(raw, int):
                data["sample_size"] = (raw, raw)
            elif isinstance(raw, (list, tuple)) and len(raw) == 2:
                data["sample_size"] = tuple(raw)
        return data


class WandBGenerateConfig(BaseModel):
    """WandB configuration for generation tracking."""

    enabled: bool = False
    project: str = "image-generation"
    entity: str | None = None
    run_name: str | None = None
    dir: str = "."
    mode: Literal["online", "offline", "disabled"] = "online"
    log_samples: bool = True  # log sample images periodically
    log_interval: int = 100  # log samples every N images


class GenerationConfig(BaseModel):
    """Root configuration for image generation."""

    seed: int = 42
    output_root: str = "data/generated"  # base output directory
    device: str = "auto"  # auto, cuda, cpu
    precision: Literal["32", "16", "bf16", "auto"] = "auto"
    models: List[ModelGenerationConfig] = Field(default_factory=list)
    wandb: WandBGenerateConfig = Field(default_factory=WandBGenerateConfig)
    save_format: Literal["png", "jpg", "npy"] = "png"  # output format
    save_individual: bool = True  # save each image individually
    save_grids: bool = True  # also save sample grids
    grid_nrow: int = 8  # number of images per row in grid
    grid_every_n: int = 64  # save a grid every N images
    auto_increment_output: bool = True  # auto-increment output dir if exists (disable for array jobs)

    @classmethod
    def load(cls, path: str | Path, overrides: Dict[str, Any] | None = None) -> "GenerationConfig":
        """Load config from YAML file with optional overrides."""
        load_dotenv()
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file '{yaml_path}' was not found.")
        yaml_data = yaml.safe_load(yaml_path.read_text()) or {}
        merged = deep_update({}, cls().model_dump())
        merged = deep_update(merged, yaml_data)
        if overrides:
            merged = deep_update(merged, overrides)
        # Interpolate ${...} references (supports array indices like ${models.0.model_type})
        merged = _interpolate_config(merged, merged)
        return cls.model_validate(merged)


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use in filenames."""
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip())
    return sanitized.strip("-_") or "model"


def _select_device(device_str: str) -> torch.device:
    """Select the appropriate device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _select_dtype(precision: str, device: torch.device) -> torch.dtype:
    """Select the appropriate dtype based on precision setting."""
    if precision == "auto":
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    precision_map = {
        "32": torch.float32,
        "16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return precision_map.get(precision, torch.float32)


def load_model_from_checkpoint(
    model_cfg: ModelGenerationConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load a model from checkpoint."""
    checkpoint_path = Path(model_cfg.checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_cls = MODEL_REGISTRY.get(model_cfg.model_type)
    if model_cls is None:
        available = ", ".join(sorted(name.value for name in MODEL_REGISTRY))
        raise ValueError(f"Unknown model type '{model_cfg.model_type}'. Available: {available}")

    # Build model kwargs
    kwargs = {
        "sample_size": model_cfg.sample_size,
        "save_dir": model_cfg.output_dir,
        **model_cfg.model_kwargs,
    }

    # Load checkpoint
    print(f"[generate] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Try to extract hyperparameters from checkpoint
            if "hyper_parameters" in checkpoint:
                hp = checkpoint["hyper_parameters"]
                # Merge with provided kwargs (provided kwargs take precedence)
                for key in ["lr", "sample_size", "num_timesteps", "latent_dim", "feature_maps"]:
                    if key in hp and key not in model_cfg.model_kwargs:
                        kwargs[key] = hp[key]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Create model instance
    print(f"[generate] Creating model instance: {model_cfg.model_type}")
    model = model_cls(**kwargs)

    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    if missing_keys:
        print(f"[generate] Warning: Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"[generate] Warning: Unexpected keys: {unexpected_keys[:5]}...")

    model = model.to(device)
    model.eval()
    print(f"[generate] Model loaded successfully on {device}")
    return model


@torch.inference_mode()
def generate_batch(
    model: torch.nn.Module,
    batch_size: int,
    model_cfg: ModelGenerationConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a batch of images from a model."""
    model_type = model_cfg.model_type

    # SMOTE is CPU-based and doesn't need autocast
    if model_type == ModelName.SMOTE:
        samples = model.sample(num_samples=batch_size)
        samples = samples.float().cpu().clamp(0, 1)
        return samples

    # Use autocast for mixed precision
    use_autocast = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
        if model_type == ModelName.GAN:
            # GAN models have a simple sample method
            samples = model.sample(
                num_samples=batch_size,
                num_inference_steps=model_cfg.num_inference_steps,
            )
        elif model_type == ModelName.GAUSSIAN_DIFFUSION:
            # Gaussian diffusion from denoising_diffusion_pytorch
            samples = model.sample(num_samples=batch_size)
        elif model_type in (ModelName.DDPM, ModelName.DDPM_V2):
            # DDPM models
            sample_kwargs: Dict[str, Any] = {"num_samples": batch_size}
            if model_cfg.num_inference_steps is not None:
                sample_kwargs["num_inference_steps"] = model_cfg.num_inference_steps
            if hasattr(model, "use_ema_for_sampling"):
                sample_kwargs["use_ema"] = model_cfg.use_ema
            samples = model.sample(**sample_kwargs)
        else:
            # Fallback: try generic sample method
            samples = model.sample(num_samples=batch_size)

    # Ensure samples are in [0, 1] range and on CPU
    samples = samples.float().cpu()
    samples = samples.clamp(0, 1)

    return samples


def save_images(
    samples: torch.Tensor,
    output_dir: Path,
    start_idx: int,
    save_format: str,
    save_individual: bool,
    save_grid: bool,
    grid_nrow: int,
) -> List[Path]:
    """Save generated images to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    if save_individual:
        import numpy as np

        for i, sample in enumerate(samples):
            idx = start_idx + i
            if save_format == "npy":
                path = output_dir / f"sample_{idx:06d}.npy"
                np.save(str(path), sample.numpy())
            else:
                path = output_dir / f"sample_{idx:06d}.{save_format}"
                tv_utils.save_image(sample, path)
            saved_paths.append(path)

    if save_grid and len(samples) > 0:
        grid = tv_utils.make_grid(samples, nrow=grid_nrow, padding=2)
        grid_path = output_dir / f"grid_{start_idx:06d}-{start_idx + len(samples) - 1:06d}.png"
        tv_utils.save_image(grid, grid_path)
        saved_paths.append(grid_path)

    return saved_paths


def generate_from_model(
    model_cfg: ModelGenerationConfig,
    output_root: Path,
    global_cfg: GenerationConfig,
    device: torch.device,
    dtype: torch.dtype,
    wandb_run: Any | None = None,
) -> Dict[str, Any]:
    """Generate all images for a single model configuration."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"[generate] Starting generation for: {model_cfg.name}")
    print(f"[generate] Model type: {model_cfg.model_type}")
    print(f"[generate] Checkpoint: {model_cfg.checkpoint_path}")
    print(f"[generate] Num images: {model_cfg.num_images}")
    print(f"[generate] Batch size: {model_cfg.batch_size}")
    print(f"{sep}\n")

    # Setup output directory (class subdirectory under the resolved output_root)
    output_dir = output_root / _sanitize_name(model_cfg.name)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite generated images: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[generate] Output directory: {output_dir}")

    # Load model
    model = load_model_from_checkpoint(model_cfg, device)

    # Generation stats
    stats = {
        "name": model_cfg.name,
        "model_type": model_cfg.model_type,
        "checkpoint": model_cfg.checkpoint_path,
        "num_images": model_cfg.num_images,
        "output_dir": str(output_dir),
        "generated_count": 0,
        "start_time": time.time(),
        "errors": [],
    }

    # Calculate number of batches
    num_batches = math.ceil(model_cfg.num_images / model_cfg.batch_size)
    generated_count = 0
    all_samples_for_final_grid: List[torch.Tensor] = []

    # Progress bar
    pbar = tqdm(total=model_cfg.num_images, desc=f"Generating {model_cfg.name}", unit="img")

    for batch_idx in range(num_batches):
        remaining = model_cfg.num_images - generated_count
        current_batch_size = min(model_cfg.batch_size, remaining)

        try:
            # Generate batch
            samples = generate_batch(model, current_batch_size, model_cfg, device, dtype)

            if len(samples) != current_batch_size:
                raise ValueError("Generator returned an unexpected batch size")

            # Save images
            save_grid = (
                global_cfg.save_grids
                and global_cfg.grid_every_n > 0  # noqa: W503
                and (generated_count % global_cfg.grid_every_n == 0 or batch_idx == num_batches - 1)  # noqa: W503
            )

            save_images(
                samples,
                output_dir,
                start_idx=generated_count,
                save_format=global_cfg.save_format,
                save_individual=global_cfg.save_individual,
                save_grid=save_grid,
                grid_nrow=global_cfg.grid_nrow,
            )

            # Keep some samples for final grid
            if len(all_samples_for_final_grid) < 64:
                all_samples_for_final_grid.append(samples[: min(8, len(samples))])

            # Log to wandb
            if wandb_run is not None and global_cfg.wandb.log_samples:
                if generated_count % global_cfg.wandb.log_interval == 0:
                    grid = tv_utils.make_grid(samples[: min(16, len(samples))], nrow=4)
                    wandb_run.log(
                        {
                            f"{model_cfg.name}/samples": wandb.Image(grid),
                            f"{model_cfg.name}/generated_count": generated_count + len(samples),
                        }
                    )

            generated_count += len(samples)
            pbar.update(len(samples))

        except Exception as e:
            error_msg = f"Error generating batch {batch_idx}: {str(e)}"
            print(f"[generate] {error_msg}")
            stats["errors"].append(error_msg)
            # Continue with next batch
            continue

    pbar.close()

    # Save final summary grid
    if global_cfg.save_grids and all_samples_for_final_grid:
        final_samples = torch.cat(all_samples_for_final_grid, dim=0)[:64]
        final_grid = tv_utils.make_grid(final_samples, nrow=8, padding=2)
        final_grid_path = output_dir / "final_summary_grid.png"
        tv_utils.save_image(final_grid, final_grid_path)
        print(f"[generate] Saved final summary grid to {final_grid_path}")

        if wandb_run is not None:
            wandb_run.log({f"{model_cfg.name}/final_grid": wandb.Image(final_grid)})

    # Update stats
    stats["generated_count"] = generated_count
    stats["end_time"] = time.time()
    stats["duration_seconds"] = stats["end_time"] - stats["start_time"]
    stats["images_per_second"] = generated_count / stats["duration_seconds"] if stats["duration_seconds"] > 0 else 0

    print(f"\n[generate] Completed {model_cfg.name}:")
    print(f"  - Generated: {generated_count}/{model_cfg.num_images} images")
    print(f"  - Duration: {stats['duration_seconds']:.2f}s")
    print(f"  - Speed: {stats['images_per_second']:.2f} img/s")

    # Clean up model to free memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return stats


def run_generation(cfg: GenerationConfig) -> Dict[str, Any]:
    """Run the full generation pipeline."""
    header = "#" * 60
    print(f"\n{header}")
    print("# Image Generation Pipeline")
    print(f"# Models to process: {len(cfg.models)}")
    print(f"# Device: {cfg.device}")
    print(f"# Precision: {cfg.precision}")
    print(f"{header}\n")

    if not cfg.models:
        raise ValueError("At least one generator model is required")
    names = [_sanitize_name(model.name) for model in cfg.models]
    if len(names) != len(set(names)):
        raise ValueError("Generator names must resolve to distinct output directories")
    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Setup device and dtype
    device = _select_device(cfg.device)
    dtype = _select_dtype(cfg.precision, device)
    print(f"[generate] Using device: {device}, dtype: {dtype}")

    # Setup WandB
    wandb_run = None
    if cfg.wandb.enabled:
        run_name = cfg.wandb.run_name or f"generation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            dir=cfg.wandb.dir,
            mode=cfg.wandb.mode,
            config=cfg.model_dump(),
        )
        print(f"[generate] WandB initialized: {run_name}")

    # Prepare output directory with auto-increment if exists (and enabled)
    base_output_root = Path(cfg.output_root).expanduser()
    output_root = base_output_root
    if cfg.auto_increment_output and output_root.exists() and any(output_root.iterdir()):
        # Directory exists and is not empty, find next available number
        counter = 1
        while True:
            output_root = Path(f"{base_output_root}_{counter}")
            if not output_root.exists() or not any(output_root.iterdir()):
                break
            counter += 1
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite generated dataset: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[generate] Output root: {output_root}")

    # Save resolved config
    config_path = output_root / "generation_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    print(f"[generate] Saved config to {config_path}")

    # Process each model
    all_stats: List[Dict[str, Any]] = []
    total_start_time = time.time()

    for model_cfg in cfg.models:
        try:
            stats = generate_from_model(model_cfg, output_root, cfg, device, dtype, wandb_run)
            all_stats.append(stats)
        except Exception as e:
            error_msg = f"Failed to process model {model_cfg.name}: {str(e)}"
            print(f"[generate] ERROR: {error_msg}")
            import traceback

            traceback.print_exc()
            all_stats.append(
                {
                    "name": model_cfg.name,
                    "error": error_msg,
                    "generated_count": 0,
                }
            )

    total_duration = time.time() - total_start_time

    # Summary
    summary = {
        "total_models": len(cfg.models),
        "successful_models": sum(1 for stats, model in zip(all_stats, cfg.models, strict=True)
                                 if stats.get("generated_count", 0) == model.num_images and "error" not in stats),
        "total_images": sum(s.get("generated_count", 0) for s in all_stats),
        "total_duration_seconds": total_duration,
        "model_stats": all_stats,
    }

    summary_sep = "=" * 60
    print(f"\n{summary_sep}")
    print("Generation Summary:")
    print(f"  - Total models: {summary['total_models']}")
    print(f"  - Successful: {summary['successful_models']}")
    print(f"  - Total images: {summary['total_images']}")
    print(f"  - Total duration: {total_duration:.2f}s")
    print(f"{summary_sep}\n")

    # Log final summary to wandb
    if wandb_run is not None:
        wandb_run.log(summary)
        wandb_run.finish()

    # Save summary to file
    summary_path = output_root / "generation_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    print(f"[generate] Saved summary to {summary_path}")

    if summary["successful_models"] != len(cfg.models):
        raise RuntimeError(f"Incomplete generation: {summary['successful_models']}/{len(cfg.models)} classes completed; see {summary_path}")
    return summary


def main(args) -> None:
    """Main entry point for generation."""

    # Load config
    cfg = GenerationConfig.load(args.config)

    # Apply CLI overrides
    if args.num_images is not None:
        for model_cfg in cfg.models:
            model_cfg.num_images = args.num_images
    if args.device is not None:
        cfg.device = args.device
    if args.output_root is not None:
        cfg.output_root = args.output_root
    if args.no_wandb:
        cfg.wandb.enabled = False

    # Select one configured model when requested
    if args.model_index is not None:
        if args.model_index < 0 or args.model_index >= len(cfg.models):
            raise ValueError(
                f"Model index {args.model_index} out of range. "
                f"Config has {len(cfg.models)} models (indices 0-{len(cfg.models) - 1})."
            )
        # Filter to only the selected model
        selected_model = cfg.models[args.model_index]
        cfg.models = [selected_model]
        # Keep the configured output path
        cfg.auto_increment_output = False
        print(f"[generate] Processing model index {args.model_index} ({selected_model.name})")
        # Append model name to wandb run name for uniqueness
        if cfg.wandb.run_name:
            cfg.wandb.run_name = f"{cfg.wandb.run_name}-{selected_model.name}"

    # Run generation
    run_generation(cfg)


