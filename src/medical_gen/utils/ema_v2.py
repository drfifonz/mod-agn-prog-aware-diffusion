"""Warmup-aware EMA utilities (v2).

This module provides an exponential moving average tracker with an inverse
decay schedule inspired by @crowsonkb's notes on long training runs. It keeps
an internal EMA copy of a model and can be swapped in place of the original
`EMA` implementation with minimal call-site changes.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Dict, Iterable, Optional, Set

import torch
from torch import Tensor, nn


def _exists(val: object) -> bool:
    return val is not None


def _is_float_tensor(tensor: Tensor) -> bool:
    return tensor.dtype in {torch.float, torch.float16, torch.bfloat16}


def _inplace_copy(dest: Tensor, src: Tensor, *, auto_move_device: bool = False) -> None:
    if auto_move_device:
        src = src.to(dest.device)
    dest.copy_(src)


def _inplace_lerp(dest: Tensor, src: Tensor, weight: float, *, auto_move_device: bool = False) -> None:
    if auto_move_device:
        src = src.to(dest.device)
    dest.lerp_(src, weight)


class EMAV2(nn.Module):
    """
    Exponential moving average tracker with inverse warmup decay.

    Parameters largely mirror `EMA` from `ema.py`, while introducing a warmup
    schedule (``inv_gamma``, ``power``) and name-based filtering to skip or copy
    specific parameters/buffers.
    """

    def __init__(
        self,
        model: nn.Module,
        ema_model: Optional[nn.Module] = None,
        *,
        beta: float = 0.9999,
        decay: Optional[float] = None,
        update_after_step: int = 100,
        update_every: int = 10,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
        param_or_buffer_names_no_ema: Iterable[str] = (),
        ignore_names: Iterable[str] = (),
        ignore_startswith_names: Iterable[str] = (),
        include_online_model: bool = True,
        allow_different_devices: bool = False,
        ema_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        chosen_beta = decay if decay is not None else beta
        if not (0.0 <= chosen_beta < 1.0):
            raise ValueError("EMA decay must be in [0, 1).")
        if update_every < 1:
            raise ValueError("update_every must be >= 1.")
        if update_after_step < 0:
            raise ValueError("update_after_step must be >= 0.")
        if power <= 0:
            raise ValueError("power must be > 0.")

        self.beta = float(chosen_beta)
        self.update_after_step = int(update_after_step)
        self.update_every = int(update_every)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.include_online_model = bool(include_online_model)
        self.allow_different_devices = bool(allow_different_devices)

        self.online_model = model if include_online_model else [model]

        if _exists(ema_model):
            self.ema_model = ema_model
        else:
            try:
                self.ema_model = deepcopy(model)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError("Failed to deepcopy model for EMA. Provide `ema_model` explicitly.") from exc

        if ema_device is not None:
            self.ema_model.to(ema_device)
        self.ema_model.requires_grad_(False)

        self.parameter_names: Set[str] = {
            name for name, param in self.ema_model.named_parameters() if _is_float_tensor(param)
        }
        self.buffer_names: Set[str] = {
            name for name, buffer in self.ema_model.named_buffers() if _is_float_tensor(buffer)
        }

        self.param_or_buffer_names_no_ema: Set[str] = set(param_or_buffer_names_no_ema)
        self.ignore_names: Set[str] = set(ignore_names)
        self.ignore_startswith_names: Set[str] = set(ignore_startswith_names)

        self.inplace_copy = partial(_inplace_copy, auto_move_device=self.allow_different_devices)
        self.inplace_lerp = partial(_inplace_lerp, auto_move_device=self.allow_different_devices)

        self._backup: Dict[str, Tensor] = {}

        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    @property
    def model(self) -> nn.Module:
        return self.online_model if self.include_online_model else self.online_model[0]

    def eval(self) -> nn.Module:  # type: ignore[override]
        return self.ema_model.eval()

    def restore_ema_model_device(self) -> None:
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
        for name, param in model.named_parameters():
            if name in self.parameter_names:
                yield name, param

    def get_buffers_iter(self, model: nn.Module) -> Iterable[tuple[str, Tensor]]:
        for name, buffer in model.named_buffers():
            if name in self.buffer_names:
                yield name, buffer

    def _should_ignore(self, name: str) -> bool:
        if name in self.ignore_names:
            return True
        return any(name.startswith(prefix) for prefix in self.ignore_startswith_names)

    def copy_params_from_model_to_ema(self, source_model: Optional[nn.Module] = None) -> None:
        copy = self.inplace_copy
        src_model = source_model if source_model is not None else self.model

        for (_, ema_param), (_, src_param) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(src_model)
        ):
            copy(ema_param.data, src_param.data)

        for (_, ema_buffer), (_, src_buffer) in zip(
            self.get_buffers_iter(self.ema_model), self.get_buffers_iter(src_model)
        ):
            copy(ema_buffer.data, src_buffer.data)

    def copy_params_from_ema_to_model(self, target_model: Optional[nn.Module] = None) -> None:
        copy = self.inplace_copy
        tgt_model = target_model if target_model is not None else self.model

        for (_, ema_param), (_, tgt_param) in zip(
            self.get_params_iter(self.ema_model), self.get_params_iter(tgt_model)
        ):
            copy(tgt_param.data, ema_param.data)

        for (_, ema_buffer), (_, tgt_buffer) in zip(
            self.get_buffers_iter(self.ema_model), self.get_buffers_iter(tgt_model)
        ):
            copy(tgt_buffer.data, ema_buffer.data)

    def get_current_decay(self) -> float:
        epoch = (self.step - self.update_after_step - 1).clamp(min=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        if epoch.item() <= 0:
            return 0.0
        return float(torch.clamp(value, min=self.min_value, max=self.beta).item())

    @torch.no_grad()
    def update(self, model: Optional[nn.Module] = None, decay: Optional[float] = None) -> None:
        current_model = model if model is not None else self.model
        if current_model is None:
            raise ValueError("No model provided for EMA update.")

        step = int(self.step.item())
        self.step += 1

        if step % self.update_every != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema(current_model)
            return

        if not bool(self.initted.item()):
            self.copy_params_from_model_to_ema(current_model)
            self.initted.data.fill_(True)

        current_decay = decay if decay is not None else self.get_current_decay()
        current_decay = float(current_decay)
        if not (0.0 <= current_decay < 1.0):
            raise ValueError("EMA decay must be in [0, 1).")
        self.update_moving_average(self.ema_model, current_model, decay=current_decay)

    @torch.no_grad()
    def update_moving_average(self, ema_model: nn.Module, current_model: nn.Module, *, decay: float) -> None:
        copy = self.inplace_copy
        lerp = self.inplace_lerp
        lerp_weight = 1.0 - decay

        for (name, current_param), (_, ema_param) in zip(
            self.get_params_iter(current_model), self.get_params_iter(ema_model)
        ):
            if self._should_ignore(name):
                continue
            if name in self.param_or_buffer_names_no_ema:
                copy(ema_param.data, current_param.data)
                continue
            lerp(ema_param.data, current_param.data, lerp_weight)

        for (name, current_buffer), (_, ema_buffer) in zip(
            self.get_buffers_iter(current_model), self.get_buffers_iter(ema_model)
        ):
            if self._should_ignore(name):
                continue
            if name in self.param_or_buffer_names_no_ema:
                copy(ema_buffer.data, current_buffer.data)
                continue
            lerp(ema_buffer.data, current_buffer.data, lerp_weight)

    @torch.no_grad()
    def copy_to(self, model: Optional[nn.Module] = None) -> None:
        target = model if model is not None else self.model
        self.copy_params_from_ema_to_model(target)

    @torch.no_grad()
    def store(self, model: Optional[nn.Module] = None) -> None:
        src_model = model if model is not None else self.model
        self._backup = {}
        for name, param in self.get_params_iter(src_model):
            self._backup[name] = param.detach().clone()
        for name, buffer in self.get_buffers_iter(src_model):
            self._backup[name] = buffer.detach().clone()

    @torch.no_grad()
    def restore(self, model: Optional[nn.Module] = None) -> None:
        if not self._backup:
            return
        tgt_model = model if model is not None else self.model
        for name, param in self.get_params_iter(tgt_model):
            if name in self._backup:
                param.data.copy_(self._backup[name].to(device=param.device, dtype=param.dtype))
        for name, buffer in self.get_buffers_iter(tgt_model):
            if name in self._backup:
                buffer.data.copy_(self._backup[name].to(device=buffer.device, dtype=buffer.dtype))
        self._backup = {}

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):  # type: ignore[override]
        if destination is None:
            from collections import OrderedDict

            destination = OrderedDict()

        for name, param in self.get_params_iter(self.ema_model):
            tensor = param if keep_vars else param.detach()
            destination[f"{prefix}{name}"] = tensor.clone()

        for name, buffer in self.get_buffers_iter(self.ema_model):
            tensor = buffer if keep_vars else buffer.detach()
            destination[f"{prefix}{name}"] = tensor.clone()

        destination[f"{prefix}__ema_step__"] = (
            self.step if keep_vars else torch.tensor(int(self.step.item()), dtype=torch.long, device=self.step.device)
        )
        destination[f"{prefix}__ema_initted__"] = (
            self.initted if keep_vars else torch.tensor(bool(self.initted.item()), device=self.initted.device)
        )

        return destination

    def load_state_dict(self, state: Dict[str, Tensor], strict: bool = True, assign: bool = False):  # type: ignore[override]
        from torch.nn.modules.module import _IncompatibleKeys

        state_copy = dict(state)
        meta_step = int(state_copy.pop("__ema_step__", torch.tensor(0)).item())
        meta_initted = bool(state_copy.pop("__ema_initted__", torch.tensor(False)).item())

        missing_keys = []
        processed: Set[str] = set()

        for name, param in self.get_params_iter(self.ema_model):
            if name in state_copy:
                param.data.copy_(state_copy[name].to(device=param.device, dtype=param.dtype))
                processed.add(name)
            else:
                missing_keys.append(name)

        for name, buffer in self.get_buffers_iter(self.ema_model):
            if name in state_copy:
                buffer.data.copy_(state_copy[name].to(device=buffer.device, dtype=buffer.dtype))
                processed.add(name)
            else:
                missing_keys.append(name)

        self.step.data.copy_(torch.tensor(meta_step, dtype=torch.long, device=self.step.device))
        self.initted.data.copy_(torch.tensor(meta_initted, device=self.initted.device))

        unexpected_keys = [key for key in state_copy.keys() if key not in processed and not key.startswith("__ema_")]

        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def sync_step(self, step: int) -> None:
        self.step.data.copy_(torch.tensor(int(step), dtype=torch.long, device=self.step.device))

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
