from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pytorch_lightning as pl
import torch
from torch import nn


def _iter_named_params(module: nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    """Yield only trainable parameters to exclude buffers from EMA tracking."""
    return ((name, param) for name, param in module.named_parameters() if param.requires_grad)


@dataclass
class EMASchedule:
    """
    Optional decay schedule.
    If you don't need scheduling, skip this and pass a fixed decay to EMA.
    """

    base_decay: float = 0.999
    warmup_steps: int = 0
    power: float = 1.0

    def __call__(self, step: int) -> float:
        decay = float(self.base_decay)
        if self.warmup_steps <= 0:
            return decay
        clamped_step = min(max(step, 0), self.warmup_steps)
        ramp = (clamped_step / self.warmup_steps) ** self.power
        return decay * ramp


class EMA:
    """
    Minimal EMA tracker for a nn.Module.

    - Keeps a shadow copy of trainable parameters.
    - `update()` applies EMA.
    - `copy_to()` loads EMA weights into a target module.
    - `store()`/`restore()` allow swapping for validation/sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = float(decay)
        self._shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self._init_from(model, device=device, dtype=dtype)

    @torch.no_grad()
    def _init_from(
        self,
        model: nn.Module,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> None:
        for name, param in _iter_named_params(model):
            tensor = param.detach()
            if device is not None:
                tensor = tensor.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            self._shadow[name] = tensor.clone()

    @torch.no_grad()
    def update(self, model: nn.Module, decay: Optional[float] = None) -> None:
        chosen_decay = self.decay if decay is None else float(decay)
        if not (0.0 < chosen_decay < 1.0):
            raise ValueError("EMA decay must be in (0, 1).")

        for name, param in _iter_named_params(model):
            if name not in self._shadow:
                self._shadow[name] = param.detach().clone()
            ema_tensor = self._shadow[name]
            param_data = param.detach()
            if param_data.device != ema_tensor.device or param_data.dtype != ema_tensor.dtype:
                param_data = param_data.to(device=ema_tensor.device, dtype=ema_tensor.dtype)
            ema_tensor.mul_(chosen_decay).add_(param_data, alpha=1.0 - chosen_decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in _iter_named_params(model):
            param.data.copy_(self._shadow[name].to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self._backup = {name: param.detach().clone() for name, param in _iter_named_params(model)}

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in _iter_named_params(model):
            param.data.copy_(self._backup[name].to(device=param.device, dtype=param.dtype))
        self._backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self._shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self._shadow = {key: value.clone() for key, value in state.items()}


class EMACallback(pl.Callback):
    """
    Lightning callback that:
    - updates EMA every `update_every` optimizer steps
    - optionally uses a decay schedule
    - swaps to EMA weights for validation (and restores after)

    Works with Hugging Face Diffusers models too, because they are plain nn.Module.
    """

    def __init__(
        self,
        decay: float = 0.999,
        update_every: int = 1,
        schedule: Optional[EMASchedule] = None,
        store_on_cpu: bool = True,
        use_ema_for_validation: bool = True,
        attr_name: str = "ema",
    ) -> None:
        if update_every < 1:
            raise ValueError("update_every must be >= 1.")
        self.decay = float(decay)
        self.update_every = int(update_every)
        self.schedule = schedule
        self.store_on_cpu = bool(store_on_cpu)
        self.use_ema_for_validation = bool(use_ema_for_validation)
        self.attr_name = str(attr_name)
        self._step: int = 0

    def _get_model(self, pl_module: pl.LightningModule) -> nn.Module:
        target = getattr(pl_module, "ema_target", None)
        return target if isinstance(target, nn.Module) else pl_module

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        target = self._get_model(pl_module)
        device = torch.device("cpu") if self.store_on_cpu else next(target.parameters()).device
        ema = EMA(target, decay=self.decay, device=device)
        setattr(pl_module, self.attr_name, ema)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self._step += 1
        if self._step % self.update_every != 0:
            return

        ema: EMA = getattr(pl_module, self.attr_name)
        target = self._get_model(pl_module)

        decay = self.schedule(self._step) if self.schedule is not None else self.decay
        ema.update(target, decay=decay)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.use_ema_for_validation:
            return
        ema: EMA = getattr(pl_module, self.attr_name)
        target = self._get_model(pl_module)
        ema.store(target)
        ema.copy_to(target)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.use_ema_for_validation:
            return
        ema: EMA = getattr(pl_module, self.attr_name)
        target = self._get_model(pl_module)
        ema.restore(target)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict,
    ) -> None:
        ema: Optional[EMA] = getattr(pl_module, self.attr_name, None)
        if ema is not None:
            checkpoint[f"{self.attr_name}_state_dict"] = ema.state_dict()

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict,
    ) -> None:
        state = checkpoint.get(f"{self.attr_name}_state_dict")
        if state is None:
            return
        ema: Optional[EMA] = getattr(pl_module, self.attr_name, None)
        if ema is not None:
            ema.load_state_dict(state)
        else:
            pl_module._pending_ema_state = state
