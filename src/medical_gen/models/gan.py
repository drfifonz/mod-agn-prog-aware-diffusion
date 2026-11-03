"""GAN-based generator using Hugging Face PreTrainedModel APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torch.nn.utils import spectral_norm
from transformers import PretrainedConfig, PreTrainedModel

from ..utils import normalize_hw
from .mixins import FIDValidationMixin, SampleVisualizationMixin

Stride = Tuple[int, int]
BASE_HW: Stride = (4, 4)


class SelfAttention(nn.Module):
    """Self-attention module for capturing long-range dependencies."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        query = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key(x).view(b, -1, h * w)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value(x).view(b, -1, h * w)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    """Residual block with spectral normalization for discriminator."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False) -> None:
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))

        self.shortcut = nn.Identity()
        if in_channels != out_channels or downsample:
            self.shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0))

        self.activation = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        return self.activation(out + residual)


class GeneratorResBlock(nn.Module):
    """Residual block for generator with upsampling."""

    def __init__(self, in_channels: int, out_channels: int, upsample: bool = False) -> None:
        super().__init__()
        self.upsample = upsample

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.upsample:
            residual = F.interpolate(residual, scale_factor=2, mode="nearest")
        residual = self.shortcut(residual)

        out = self.bn1(x)
        out = self.activation(out)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        return out + residual


def _kernel_for_stride(stride: Stride) -> Stride:
    return tuple(4 if dim > 1 else 3 for dim in stride)


def _compute_stride_plan(target_hw: Stride) -> List[Stride]:
    target_h, target_w = target_hw
    if target_h < BASE_HW[0] or target_w < BASE_HW[1]:
        raise ValueError("GAN sample_size must be at least 4x4.")
    current_h, current_w = BASE_HW
    plan: List[Stride] = []
    while current_h < target_h or current_w < target_w:
        stride_h = 2 if current_h < target_h else 1
        stride_w = 2 if current_w < target_w else 1
        next_h = current_h * stride_h
        next_w = current_w * stride_w
        if next_h > target_h or next_w > target_w:
            raise ValueError(
                "GAN sample_size must be multiples of powers of two relative to 4x4 (e.g., 64x32, 256x128)."
            )
        plan.append((stride_h, stride_w))
        current_h, current_w = next_h, next_w
    if (current_h, current_w) != target_hw:
        raise ValueError("Failed to build stride plan for requested GAN sample_size.")
    return plan


def _build_generator_layers(
    config: "HFGANGeneratorConfig",
    plan: Sequence[Stride],
) -> nn.Sequential:
    fm = config.feature_maps
    layers: list[nn.Module] = [
        nn.ConvTranspose2d(config.latent_dim, fm * 4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(fm * 4),
        nn.ReLU(True),
    ]
    if not plan:
        layers.extend(
            [
                nn.ConvTranspose2d(fm * 4, config.image_channels, 3, 1, 1, bias=False),
                nn.Tanh(),
            ]
        )
        return nn.Sequential(*layers)

    in_channels = fm * 4
    for idx, stride in enumerate(plan):
        is_last = idx == len(plan) - 1
        out_channels = config.image_channels if is_last else max(config.image_channels, in_channels // 2)
        kernel_size = _kernel_for_stride(stride)
        padding = (1, 1)
        layers.append(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
        )
        if is_last:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*layers)


class ImprovedGenerator(nn.Module):
    """Improved generator with residual blocks and self-attention for high-quality image synthesis."""

    def __init__(
        self,
        latent_dim: int = 256,
        image_channels: int = 1,
        feature_maps: int = 64,
        image_size: Stride = (128, 256),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # For 128x256: start from 4x8, upsample 5 times (4->8->16->32->64->128 for H, 8->16->32->64->128->256 for W)
        # Initial projection: latent -> 4x8 spatial
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fm := feature_maps * 16, (4, 8), 1, 0, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(True),
        )

        # Upsampling blocks with residual connections
        self.block1 = GeneratorResBlock(fm, fm, upsample=True)  # 4x8 -> 8x16
        self.block2 = GeneratorResBlock(fm, fm // 2, upsample=True)  # 8x16 -> 16x32
        self.block3 = GeneratorResBlock(fm // 2, fm // 4, upsample=True)  # 16x32 -> 32x64
        self.attn = SelfAttention(fm // 4)  # Self-attention at 32x64 resolution
        self.block4 = GeneratorResBlock(fm // 4, fm // 8, upsample=True)  # 32x64 -> 64x128
        self.block5 = GeneratorResBlock(fm // 8, fm // 16, upsample=True)  # 64x128 -> 128x256

        self.final = nn.Sequential(
            nn.BatchNorm2d(fm // 16),
            nn.ReLU(True),
            nn.Conv2d(fm // 16, image_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.initial(z)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.attn(out)
        out = self.block4(out)
        out = self.block5(out)
        return self.final(out)


class ImprovedDiscriminator(nn.Module):
    """Improved discriminator with spectral normalization and self-attention."""

    def __init__(
        self,
        image_channels: int = 1,
        feature_maps: int = 64,
        image_size: Stride = (128, 256),
    ) -> None:
        super().__init__()
        fm = feature_maps

        # Input: 128x256
        self.initial = spectral_norm(nn.Conv2d(image_channels, fm, 4, 2, 1))  # -> 64x128

        self.block1 = ResidualBlock(fm, fm * 2, downsample=True)  # -> 32x64
        self.block2 = ResidualBlock(fm * 2, fm * 4, downsample=True)  # -> 16x32
        self.attn = SelfAttention(fm * 4)  # Self-attention at 16x32
        self.block3 = ResidualBlock(fm * 4, fm * 8, downsample=True)  # -> 8x16
        self.block4 = ResidualBlock(fm * 8, fm * 16, downsample=True)  # -> 4x8

        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(fm * 16, 1, (4, 8), 1, 0)),  # -> 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.initial(x), 0.2)
        out = self.block1(out)
        out = self.block2(out)
        out = self.attn(out)
        out = self.block3(out)
        out = self.block4(out)
        return self.final(out)


class HFGANGeneratorConfig(PretrainedConfig):
    model_type = "hf_gan_generator"

    def __init__(
        self,
        latent_dim: int = 128,
        image_channels: int = 1,
        feature_maps: int = 64,
        image_size: Stride = (32, 32),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.feature_maps = feature_maps
        self.image_size = normalize_hw(image_size, field_name="gan.image_size")


class HFGANGenerator(PreTrainedModel):
    config_class = HFGANGeneratorConfig

    def __init__(self, config: HFGANGeneratorConfig) -> None:
        super().__init__(config)
        plan = _compute_stride_plan(tuple(config.image_size))
        self.net = _build_generator_layers(config, plan)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(latent)


class GANGenerator(FIDValidationMixin, SampleVisualizationMixin, pl.LightningModule):
    """Improved GAN with modern training techniques for high-quality image synthesis."""

    def __init__(
        self,
        lr: float = 2e-4,
        lr_d: float | None = None,  # Discriminator LR (TTUR: typically 4x generator LR)
        sample_size: Stride = (32, 32),
        num_timesteps: int | None = None,
        save_dir: str = "samples",
        sampling_every_n_steps: int = 0,
        sampling_num_images: int = 16,
        sampling_num_inference_steps: int = 50,
        sampling_grid_nrow: int = 4,
        fid_num_samples: int = 256,
        fid_enabled: bool = True,
        fid_low_vram_mode: bool = False,
        fid_clear_cuda_cache: bool = True,
        latent_dim: int = 256,
        feature_maps: int = 64,
        use_improved_arch: bool = True,  # Use new improved architecture
        use_hinge_loss: bool = True,  # Hinge loss instead of BCE
        label_smoothing: float = 0.1,  # Label smoothing for stability
        n_critic: int = 1,  # Discriminator updates per generator update
        **_: Any,
    ) -> None:
        pl.LightningModule.__init__(self)
        self.automatic_optimization = False
        FIDValidationMixin.__init__(
            self,
            fid_num_samples=fid_num_samples,
            low_vram_mode=fid_low_vram_mode,
            clear_cuda_cache=fid_clear_cuda_cache,
        )
        self.fid_enabled = fid_enabled
        self.sample_size = normalize_hw(sample_size, field_name="model.sample_size")
        self._stride_plan = _compute_stride_plan(self.sample_size)
        self.lr = lr
        self.lr_d = lr_d if lr_d is not None else lr * 4  # TTUR default
        self.save_dir = Path(save_dir)
        self.sampling_every_n_steps = sampling_every_n_steps
        self.sampling_num_images = sampling_num_images
        self.sampling_num_inference_steps = sampling_num_inference_steps
        self.sampling_grid_nrow = sampling_grid_nrow
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.use_hinge_loss = use_hinge_loss
        self.label_smoothing = label_smoothing
        self.n_critic = n_critic
        self.use_improved_arch = use_improved_arch

        # Check if using 128x256 resolution for improved architecture
        is_128x256 = self.sample_size == (128, 256)

        if use_improved_arch and is_128x256:
            # Use improved architecture for 128x256 images
            self.generator = ImprovedGenerator(
                latent_dim=latent_dim,
                image_channels=1,
                feature_maps=feature_maps,
                image_size=self.sample_size,
            )
            self.discriminator = ImprovedDiscriminator(
                image_channels=1,
                feature_maps=feature_maps,
                image_size=self.sample_size,
            )
        else:
            # Fall back to original architecture for other sizes
            self.generator = HFGANGenerator(
                HFGANGeneratorConfig(
                    latent_dim=latent_dim,
                    image_channels=1,
                    feature_maps=feature_maps,
                    image_size=self.sample_size,
                )
            )
            self.discriminator = self._build_discriminator(feature_maps, self._stride_plan)

        # Loss function
        if not use_hinge_loss:
            self.criterion = nn.BCEWithLogitsLoss()

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization for better training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_discriminator(self, feature_maps: int, plan: Sequence[Stride]) -> nn.Sequential:
        """Original discriminator architecture with spectral normalization."""
        channel_schedule = [feature_maps, feature_maps * 2, feature_maps * 4]
        layers: list[nn.Module] = []
        in_channels = 1
        for idx, stride in enumerate(reversed(plan)):
            out_channels = channel_schedule[min(idx, len(channel_schedule) - 1)]
            kernel_size = _kernel_for_stride(stride)
            padding = (1, 1)
            layers.append(
                spectral_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                )
            )
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            in_channels = out_channels
        layers.append(spectral_norm(nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False)))
        return nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.generator(latent)

    def _hinge_loss_d(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Hinge loss for discriminator."""
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return real_loss + fake_loss

    def _hinge_loss_g(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Hinge loss for generator."""
        return -fake_logits.mean()

    def _bce_loss_d(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """BCE loss with label smoothing for discriminator."""
        real_targets = torch.ones_like(real_logits) * (1.0 - self.label_smoothing)
        fake_targets = torch.zeros_like(fake_logits) + self.label_smoothing * 0.5
        real_loss = self.criterion(real_logits, real_targets)
        fake_loss = self.criterion(fake_logits, fake_targets)
        return 0.5 * (real_loss + fake_loss)

    def _bce_loss_g(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """BCE loss for generator."""
        valid_targets = torch.ones_like(fake_logits)
        return self.criterion(fake_logits, valid_targets)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        images, _ = batch
        batch_size = images.size(0)
        opt_d, opt_g = self.optimizers()

        # --- discriminator update(s) ---
        for _ in range(self.n_critic):
            self.toggle_optimizer(opt_d)
            opt_d.zero_grad(set_to_none=True)

            z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            with torch.no_grad():
                fake_imgs = self.generator(z)

            real_logits = self.discriminator(images)
            fake_logits = self.discriminator(fake_imgs)

            if self.use_hinge_loss:
                d_loss = self._hinge_loss_d(real_logits, fake_logits)
            else:
                d_loss = self._bce_loss_d(real_logits, fake_logits)

            self.manual_backward(d_loss)
            opt_d.step()
            self.untoggle_optimizer(opt_d)

        self.log("disc_loss", d_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        # --- generator update ---
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad(set_to_none=True)

        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        gen_imgs = self.generator(z)
        fake_logits = self.discriminator(gen_imgs)

        if self.use_hinge_loss:
            g_loss = self._hinge_loss_g(fake_logits)
        else:
            g_loss = self._bce_loss_g(fake_logits)

        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.log("gen_loss", g_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        # Two-Timescale Update Rule (TTUR): different LRs for G and D
        betas = (0.0, 0.9)  # Better for spectral norm + hinge loss
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=betas)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=betas)
        return [opt_d, opt_g]

    # --- FID integration -------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        if not self.fid_enabled:
            return
        FIDValidationMixin.on_validation_epoch_start(self)

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        if not self.fid_enabled:
            return
        FIDValidationMixin.validation_step(self, batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        if not self.fid_enabled:
            return
        FIDValidationMixin.on_validation_epoch_end(self)

    @torch.no_grad()
    def sample(self, num_samples: int = 16, num_inference_steps: int | None = None) -> torch.Tensor:
        was_training = self.generator.training
        self.generator.eval()
        latent = torch.randn(num_samples, self.latent_dim, 1, 1, device=self.device)
        samples = self.generator(latent).cpu()
        if was_training:
            self.generator.train()
        samples = (samples + 1) / 2.0
        return samples

    def on_train_end(self) -> None:
        self.print("[on_train_end] Generating final samples...")
        samples = self.sample(num_samples=self.sampling_num_images)
        grid, grid_path = self._save_samples_grid(
            samples,
            step=None,
            prefix="gan_samples",
        )
        self.print(f"[on_train_end] Saved samples to {grid_path}")
        logger = self.logger
        if isinstance(logger, WandbLogger):
            self.print("[on_train_end] Logging final samples to WandB...")
            logger.log_image(
                key="gan_final_samples",
                images=[grid],
                caption=["training_end"],
            )
            self.print("[on_train_end] WandB image logged.")
        self.print("[on_train_end] Complete.")

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.sampling_every_n_steps <= 0:
            return
        if self.global_step == 0 or self.global_step % self.sampling_every_n_steps != 0:
            return
        if getattr(self, "global_rank", 0) != 0:
            return
        samples = self.sample(num_samples=self.sampling_num_images)
        grid, grid_path = self._save_samples_grid(
            samples,
            step=self.global_step,
            prefix="gan_samples",
        )
        self.print(f"Saved intermediate GAN samples to {grid_path}")
        logger = self.logger
        if isinstance(logger, WandbLogger):
            logger.log_image(
                key="gan_generated_samples",
                images=[grid],
                caption=[f"step {self.global_step}"],
            )

    def generate_samples_for_fid(self, num_samples: int) -> torch.Tensor:
        samples = self.sample(num_samples=num_samples)
        return samples
