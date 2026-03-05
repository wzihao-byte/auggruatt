from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from PrunedAttentionGRU import TEMPORAL_POOLING_MODES, prunedAttentionGRU


MODEL_VARIANTS = ("baseline", "dual_concat", "dual_gated")
TEMPORAL_POOLING_CHOICES = TEMPORAL_POOLING_MODES


def _reshape_sequence(x: torch.Tensor) -> torch.Tensor:
    """
    Convert model input to [B, T, D] for frequency extraction.
    """
    if x.dim() < 2:
        raise ValueError(f"expected at least 2D input, got shape={tuple(x.shape)}")
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() == 3:
        return x
    batch_size, time_steps = x.shape[0], x.shape[1]
    return x.reshape(batch_size, time_steps, -1)


def extract_frequency_signature(
    x: torch.Tensor,
    use_abs: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build deterministic frequency signature from input sequence.

    Steps:
    1) Per-frame summary: a_t = mean_d(|x_t,d|) if use_abs else mean_d(x_t,d)
    2) Demean across time
    3) rFFT over time axis and magnitude
    4) L2 normalization for stability

    Note:
        In this repository, model input is typically standardized real-valued features.
        For that case, use_abs=False is the safer default.

    Input:
        x: [B, T, D] (or compatible sequence layout)
    Output:
        signature: [B, floor(T/2) + 1]
    """
    seq = _reshape_sequence(x)
    if use_abs:
        seq = seq.abs()

    amplitude = seq.mean(dim=-1)
    amplitude = amplitude - amplitude.mean(dim=1, keepdim=True)

    spectrum = torch.fft.rfft(amplitude, dim=1)
    signature = spectrum.abs()

    denom = signature.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    signature = signature / denom
    return signature


class FrequencyBranch(nn.Module):
    """
    Lightweight MLP branch for the frequency signature.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        use_abs: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.use_abs = bool(use_abs)
        self.eps = float(eps)
        self.last_signature_dim: Optional[int] = None

        self.proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        signature = extract_frequency_signature(x=x, use_abs=self.use_abs, eps=self.eps)
        self.last_signature_dim = int(signature.shape[-1])
        feature = self.proj(signature)
        return feature, signature


class DualConcatPrunedAttentionGRU(nn.Module):
    """
    Dual-domain model with concatenation fusion.

    - Time branch: existing GRU + attention feature path
    - Frequency branch: lightweight MLP over deterministic frequency signature
    - Fusion head: concat(time, freq) -> MLP -> logits
    """

    model_variant = "dual_concat"
    fusion_type = "concat"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attention_dim: int,
        output_dim: int,
        freq_feature_dim: int = 64,
        fusion_hidden_dim: int = 64,
        freq_use_abs: bool = False,
        freq_eps: float = 1e-8,
        temporal_pooling: str = "attn",
    ) -> None:
        super().__init__()
        self.time_encoder = prunedAttentionGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            output_dim=None,
            temporal_pooling=temporal_pooling,
        )
        self.temporal_pooling = self.time_encoder.temporal_pooling
        self.freq_branch = FrequencyBranch(
            feature_dim=freq_feature_dim,
            hidden_dim=freq_feature_dim,
            use_abs=freq_use_abs,
            eps=freq_eps,
        )
        self.freq_use_abs = bool(freq_use_abs)
        self.fusion_head = nn.Sequential(
            nn.Linear(attention_dim + freq_feature_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

    @property
    def frequency_signature_dim(self) -> Optional[int]:
        return self.freq_branch.last_signature_dim

    def forward_with_aux(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if hasattr(self.time_encoder, "forward_with_aux"):
            time_feat, time_aux = self.time_encoder.forward_with_aux(x)  # type: ignore[attr-defined]
        else:
            time_feat = self.time_encoder.extract_time_features(x)
            time_aux = {}
        freq_feat, freq_signature = self.freq_branch(x)
        fused = torch.cat([time_feat, freq_feat], dim=1)
        logits = self.fusion_head(fused)
        aux = {
            "time_feat": time_feat,
            "freq_feat": freq_feat,
            "freq_signature": freq_signature,
            "time_aux": time_aux,
            "attention_weights": time_aux.get("attention_weights"),
        }
        return logits, aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_aux(x)
        return logits


class DualGatedPrunedAttentionGRU(nn.Module):
    """
    Dual-domain model with gated fusion.

    Gate design:
        g = sigmoid(MLP([time_feat, freq_feat]))
        fused = concat(time_feat, g * freq_feat)
    """

    model_variant = "dual_gated"
    fusion_type = "gated"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attention_dim: int,
        output_dim: int,
        freq_feature_dim: int = 64,
        fusion_hidden_dim: int = 64,
        freq_use_abs: bool = False,
        freq_eps: float = 1e-8,
        temporal_pooling: str = "attn",
    ) -> None:
        super().__init__()
        self.time_encoder = prunedAttentionGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            output_dim=None,
            temporal_pooling=temporal_pooling,
        )
        self.temporal_pooling = self.time_encoder.temporal_pooling
        self.freq_branch = FrequencyBranch(
            feature_dim=freq_feature_dim,
            hidden_dim=freq_feature_dim,
            use_abs=freq_use_abs,
            eps=freq_eps,
        )
        self.freq_use_abs = bool(freq_use_abs)
        self.gate_mlp = nn.Sequential(
            nn.Linear(attention_dim + freq_feature_dim, freq_feature_dim),
            nn.Sigmoid(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(attention_dim + freq_feature_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

    @property
    def frequency_signature_dim(self) -> Optional[int]:
        return self.freq_branch.last_signature_dim

    def forward_with_aux(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if hasattr(self.time_encoder, "forward_with_aux"):
            time_feat, time_aux = self.time_encoder.forward_with_aux(x)  # type: ignore[attr-defined]
        else:
            time_feat = self.time_encoder.extract_time_features(x)
            time_aux = {}
        freq_feat, freq_signature = self.freq_branch(x)
        gate_input = torch.cat([time_feat, freq_feat], dim=1)
        gate = self.gate_mlp(gate_input)
        fused = torch.cat([time_feat, gate * freq_feat], dim=1)
        logits = self.fusion_head(fused)
        aux = {
            "time_feat": time_feat,
            "freq_feat": freq_feat,
            "freq_signature": freq_signature,
            "gate": gate,
            "time_aux": time_aux,
            "attention_weights": time_aux.get("attention_weights"),
        }
        return logits, aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_aux(x)
        return logits


def build_model(
    model_variant: str,
    input_dim: int,
    hidden_dim: int,
    attention_dim: int,
    output_dim: int,
    freq_feature_dim: int = 64,
    fusion_hidden_dim: int = 64,
    freq_use_abs: bool = False,
    freq_eps: float = 1e-8,
    temporal_pooling: str = "attn",
) -> nn.Module:
    variant = str(model_variant).strip().lower()
    if variant == "baseline":
        model = prunedAttentionGRU(
            input_dim,
            hidden_dim,
            attention_dim,
            output_dim,
            temporal_pooling=temporal_pooling,
        )
        setattr(model, "model_variant", "baseline")
        setattr(model, "fusion_type", "none")
        setattr(model, "temporal_pooling", model.temporal_pooling)
        return model
    if variant == "dual_concat":
        return DualConcatPrunedAttentionGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            output_dim=output_dim,
            freq_feature_dim=freq_feature_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            freq_use_abs=freq_use_abs,
            freq_eps=freq_eps,
            temporal_pooling=temporal_pooling,
        )
    if variant == "dual_gated":
        return DualGatedPrunedAttentionGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            output_dim=output_dim,
            freq_feature_dim=freq_feature_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            freq_use_abs=freq_use_abs,
            freq_eps=freq_eps,
            temporal_pooling=temporal_pooling,
        )
    raise ValueError(f"unsupported model_variant={model_variant!r}; expected one of {MODEL_VARIANTS}")


def model_metadata(model: nn.Module) -> Dict[str, Any]:
    projected_dim = getattr(getattr(model, "freq_branch", None), "feature_dim", None)
    temporal_pooling = getattr(model, "temporal_pooling", None)
    if temporal_pooling is None and hasattr(model, "time_encoder"):
        temporal_pooling = getattr(model.time_encoder, "temporal_pooling", None)
    return {
        "model_variant": str(getattr(model, "model_variant", "baseline")),
        "fusion_type": str(getattr(model, "fusion_type", "none")),
        "temporal_pooling": temporal_pooling,
        "frequency_use_abs": getattr(model, "freq_use_abs", None),
        "raw_frequency_signature_dim": getattr(model, "frequency_signature_dim", None),
        "projected_frequency_feature_dim": projected_dim,
    }
