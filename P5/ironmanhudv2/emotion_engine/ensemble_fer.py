from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleFER(nn.Module):
    """Probability-averaging ensemble wrapper over multiple FER submodels."""

    def __init__(self, submodels: Sequence[nn.Module]):
        super().__init__()
        submodel_list = list(submodels)
        if not submodel_list:
            raise ValueError("EnsembleFER requires at least one submodel.")
        self.submodels = nn.ModuleList(submodel_list)

    def forward_probs(self, x: torch.Tensor, use_tta: bool = True) -> torch.Tensor:
        probs_sum: torch.Tensor | None = None
        for net in self.submodels:
            probs = F.softmax(net(x), dim=-1)
            if use_tta:
                probs_flip = F.softmax(net(torch.flip(x, dims=[-1])), dim=-1)
                probs = 0.5 * (probs + probs_flip)
            probs_sum = probs if probs_sum is None else (probs_sum + probs)
        return probs_sum / len(self.submodels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.forward_probs(x, use_tta=False)
        return torch.log(probs.clamp_min(1e-8))
