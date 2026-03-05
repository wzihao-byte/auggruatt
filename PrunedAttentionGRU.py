import torch.nn as nn
import torch
import numpy as np
import random

from MaskedAttention import MaskedAttention, MaskedLinear
from prunedGRU import CustomGRU


TEMPORAL_POOLING_MODES = ("attn", "mean", "last")


class PruningModule(nn.Module):
    def prune_by_std(self, s, k): #가중치의 절대값 표준편차를 기준으로 pruning
        #s 표준편차의 임계값을 설정하는데 사용(e.g. s=0.5, std의 절반을 임계값으로 사용)
        #k pruning ratio를 설정. 
        for name, module in self.named_modules():
            if isinstance(module, (MaskedLinear, CustomGRU, MaskedAttention)):
                self._prune_weights(module, s, k, name)

    def _prune_weights(self, module, s, k, name): 
        for attr_name in ['weight_ih', 'weight_hh', 'weight']:
            if hasattr(module, attr_name):
                weight = getattr(module, attr_name)
                threshold = np.std(weight.data.abs().cpu().numpy()) * s
                print(f'Pruning {attr_name} with threshold: {threshold} for layer {name}')
                while not module.prune(threshold, k):
                    threshold *= 0.99

    def prune_by_random(self, connectivity): #무작위로 pruning
        for name, module in self.named_modules():
            if isinstance(module, (MaskedLinear, CustomGRU, MaskedAttention)):
                self._random_prune_weights(module, connectivity, name)

    def _random_prune_weights(self, module, connectivity, name):
        for attr_name in ['weight_ih', 'weight_hh', 'weight']:
            if hasattr(module, attr_name):
                weight = getattr(module, attr_name)
                print(f'Pruning {attr_name} randomly for layer {name}')
                row = weight.shape[0]
                column = weight.shape[1]
                weight_mask = torch.tensor(self.generate_weight_mask((row, column), connectivity)).float()
                weight_data = nn.init.orthogonal_(weight.data)
                weight_data = weight_data * weight_mask
                weight.data = weight_data

    def generate_weight_mask(self, shape, connection):
        sub_shape = (shape[0], shape[1])
        w = []
        w.append(self.generate_mask_matrix(sub_shape, connection))
        return w[0]

    @staticmethod
    def generate_mask_matrix(shape, connection): #connection으로 설정된 비율에 따라 mask 행렬 생성. 
        random.seed(1)
        s = np.random.uniform(size=shape)
        s_flat = s.flatten()
        s_flat.sort()
        threshold = s_flat[int(shape[0] * shape[1] * (1 - connection))]
        super_threshold_indices = s >= threshold
        lower_threshold_indices = s < threshold
        s[super_threshold_indices] = 1.
        s[lower_threshold_indices] = 0.
        return s
    
# Define the AttentionMaskedGRU
class prunedAttentionGRU(PruningModule):
    """
    Lightweight Attention-GRU baseline.

    Input shape:
        x: [B, T, D]
    Output shape:
        logits: [B, output_dim] (when classifier is enabled)
    """

    def __init__(self, input_dim, hidden_dim, attention_dim, output_dim=None, temporal_pooling="attn"):
        super(prunedAttentionGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.temporal_pooling = str(temporal_pooling).strip().lower()
        if self.temporal_pooling not in TEMPORAL_POOLING_MODES:
            raise ValueError(f"unsupported temporal_pooling={temporal_pooling!r}; expected one of {TEMPORAL_POOLING_MODES}")
        self.gru = CustomGRU(input_dim, hidden_dim, bias=True, batch_first=True)
        self.attention = MaskedAttention(hidden_dim, attention_dim) if self.temporal_pooling == "attn" else None
        self.temporal_projection = (
            MaskedLinear(hidden_dim, attention_dim) if self.temporal_pooling in {"mean", "last"} else None
        )
        self.fc = MaskedLinear(attention_dim, output_dim) if output_dim is not None else None

    def _pool_hidden_states(self, gru_out, attention_override=None):
        if self.temporal_pooling == "attn":
            if self.attention is None:
                raise RuntimeError("attention module is not initialized")
            context_vector, attn_weights = self.attention(
                gru_out,
                return_weights=True,
                attention_override=attention_override,
            )
            return context_vector, attn_weights

        if self.temporal_pooling == "mean":
            pooled = gru_out.mean(dim=1)
            if self.temporal_projection is None:
                raise RuntimeError("temporal projection is not initialized for mean pooling")
            return self.temporal_projection(pooled), None

        if self.temporal_pooling == "last":
            pooled = gru_out[:, -1, :]
            if self.temporal_projection is None:
                raise RuntimeError("temporal projection is not initialized for last pooling")
            return self.temporal_projection(pooled), None

        raise ValueError(f"unsupported temporal_pooling={self.temporal_pooling}")

    def extract_time_features(self, x):
        """
        Return the pre-classifier time-domain feature.

        Returns:
            context_vector: [B, attention_dim]
        """
        gru_out, _ = self.gru(x)
        context_vector, _ = self._pool_hidden_states(gru_out)
        return context_vector

    def forward_with_aux(self, x, attention_override=None):
        gru_out, _ = self.gru(x)
        context_vector, attn_weights = self._pool_hidden_states(gru_out, attention_override=attention_override)

        if self.fc is None:
            output = context_vector
        else:
            output = self.fc(context_vector)

        return output, {
            "temporal_pooling": self.temporal_pooling,
            "time_feat": context_vector,
            "attention_weights": attn_weights,
        }

    def forward(self, x, return_features=False):
        output, aux = self.forward_with_aux(x)
        if return_features:
            return output, aux["time_feat"]
        return output
    
