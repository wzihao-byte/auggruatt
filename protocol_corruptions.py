from typing import Dict, List

import numpy as np
import torch


SUPPORTED_CORRUPTIONS = (
    "gaussian_noise",
    "frame_dropout",
    "subcarrier_dropout",
    "temporal_shift",
)


DEFAULT_SEVERITY_LEVELS: Dict[str, List[float]] = {
    "gaussian_noise": [0.01, 0.03, 0.05, 0.08, 0.12],
    "frame_dropout": [0.05, 0.10, 0.15, 0.25, 0.35],
    "subcarrier_dropout": [0.05, 0.10, 0.15, 0.25, 0.35],
    "temporal_shift": [0.02, 0.05, 0.10, 0.15, 0.20],
}


def resolve_severity_levels(corruption: str, override_levels: List[float]) -> List[float]:
    if override_levels:
        return [float(v) for v in override_levels]
    return DEFAULT_SEVERITY_LEVELS[corruption]


def apply_corruption(inputs: torch.Tensor, corruption: str, severity: float, seed: int) -> torch.Tensor:
    if corruption not in SUPPORTED_CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}")

    rng = np.random.RandomState(int(seed))
    if corruption == "gaussian_noise":
        return _gaussian_noise(inputs, severity, rng)
    if corruption == "frame_dropout":
        return _frame_dropout(inputs, severity, rng)
    if corruption == "subcarrier_dropout":
        return _subcarrier_dropout(inputs, severity, rng)
    if corruption == "temporal_shift":
        return _temporal_shift(inputs, severity, rng)
    return inputs


def _gaussian_noise(inputs: torch.Tensor, severity: float, rng: np.random.RandomState) -> torch.Tensor:
    batch = inputs.shape[0]
    flat = inputs.view(batch, -1).float()
    scale = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
    view_shape = [batch] + [1] * (inputs.ndim - 1)
    scale = scale.view(*view_shape)

    noise_np = rng.normal(0.0, 1.0, size=tuple(inputs.shape)).astype(np.float32)
    noise = torch.from_numpy(noise_np).to(inputs.device)
    return inputs + noise * scale * float(severity)


def _frame_dropout(inputs: torch.Tensor, severity: float, rng: np.random.RandomState) -> torch.Tensor:
    if inputs.ndim < 2:
        return inputs
    rate = float(np.clip(severity, 0.0, 1.0))
    t = inputs.shape[1]
    mask_np = (rng.rand(inputs.shape[0], t) >= rate).astype(np.float32)
    mask = torch.from_numpy(mask_np).to(inputs.device)
    shape = [inputs.shape[0], t] + [1] * (inputs.ndim - 2)
    return inputs * mask.view(*shape)


def _subcarrier_dropout(inputs: torch.Tensor, severity: float, rng: np.random.RandomState) -> torch.Tensor:
    if inputs.ndim < 3:
        return inputs
    rate = float(np.clip(severity, 0.0, 1.0))
    f = inputs.shape[2]
    mask_np = (rng.rand(inputs.shape[0], f) >= rate).astype(np.float32)
    mask = torch.from_numpy(mask_np).to(inputs.device)
    shape = [inputs.shape[0], 1, f] + [1] * (inputs.ndim - 3)
    return inputs * mask.view(*shape)


def _temporal_shift(inputs: torch.Tensor, severity: float, rng: np.random.RandomState) -> torch.Tensor:
    if inputs.ndim < 2:
        return inputs
    t = inputs.shape[1]
    max_shift = int(round(float(severity) * t))
    if max_shift <= 0:
        return inputs

    out = torch.zeros_like(inputs)
    shifts = rng.randint(-max_shift, max_shift + 1, size=inputs.shape[0])
    for i, shift in enumerate(shifts):
        if shift == 0:
            out[i] = inputs[i]
        elif shift > 0:
            out[i, shift:, ...] = inputs[i, : t - shift, ...]
        else:
            s = abs(int(shift))
            out[i, : t - s, ...] = inputs[i, s:, ...]
    return out
