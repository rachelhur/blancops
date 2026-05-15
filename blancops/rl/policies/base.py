"""Base classes and shared metric utilities for policies.

Two separate ABCs:
  * `PolicyBase` — used by BC. Owns a network and produces a loss.
  * `QAdapterBase` — used by value-based algorithms. Adapts a network's
    output to a flat Q-value tensor.

Shared metric computations are free functions so either base can use them
without forcing a common parent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from blancops.configs.constants import FILTER2IDX
from blancops.math import geometry

class PolicyBase(nn.Module, ABC):
    @abstractmethod
    def select_action(self, x_glob, x_bin, action_mask=None) -> torch.Tensor:
        ...

    @staticmethod
    def compute_heavy_metrics(
        predicted_actions: torch.Tensor,
        expert_actions: torch.Tensor,
        hpGrid,
        num_filters: int | None,
    ) -> dict:
        """Domain-specific metrics: angular separation, bin/filter accuracy,
        prediction diversity. Pulled out so Q-adapters can call it too."""
        if hpGrid is None:
            return {}

        predicted_actions = predicted_actions.detach().cpu()
        expert_actions = expert_actions.detach().cpu()

        metrics: dict = {
            "accuracy": (predicted_actions == expert_actions).float().mean().item(),
        }

        if num_filters is not None and num_filters != 1:
            predicted_bins = predicted_actions // num_filters
            expert_bins = expert_actions // num_filters
            predicted_filters = predicted_actions % num_filters
            expert_filters = expert_actions % num_filters

            metrics["filter_accuracy"] = (predicted_filters == expert_filters).float().mean().item()
            metrics["unique_filters"] = len(torch.unique(predicted_filters)) / num_filters
        else:
            predicted_bins = predicted_actions
            expert_bins = expert_actions

        metrics["bin_accuracy"] = (predicted_bins == expert_bins).float().mean().item()
        metrics["ang_sep"] = compute_slew_distance(predicted_bins, expert_bins, hpGrid)
        metrics["unique_bins"] = len(torch.unique(predicted_bins)) / len(hpGrid.lon)
        return metrics
    
class BCPolicyBase(PolicyBase, ABC):
    """A loss strategy for BC. Wraps a network and defines how its outputs
    map to a loss against expert actions."""

    def compute_standard_metrics(
        self,
        q_vals_or_logits: torch.Tensor,
        expert_flat: torch.Tensor,
        action_masks: torch.Tensor,
        num_filters: int | None = None,
        hpGrid=None,
    ) -> dict:
        """Entropy, margin, and (if hpGrid is given) angular-separation metrics.

        Used by any policy that produces flat `(batch, num_actions)` logits.
        """
        metrics: dict = {}
        mask_val = torch.finfo(q_vals_or_logits.dtype).min

        with torch.no_grad():
            masked_logits = q_vals_or_logits.masked_fill(~action_masks, mask_val)
            pred_actions = masked_logits.argmax(dim=1)

            logp = F.log_softmax(q_vals_or_logits, dim=-1)
            logp_expert = logp.gather(1, expert_flat.unsqueeze(1)).squeeze(1)
            p = F.softmax(q_vals_or_logits, dim=-1)
            entropy = -(p * logp).sum(dim=-1).mean().item()

            num_actions = q_vals_or_logits.size(1)
            z_expert = q_vals_or_logits.gather(1, expert_flat.unsqueeze(1)).squeeze(1)
            expert_mask = F.one_hot(expert_flat, num_classes=num_actions).bool()
            z_max_other = q_vals_or_logits.masked_fill(expert_mask, float("-inf")).max(dim=1).values
            margin = (z_expert - z_max_other).mean().item()

            metrics.update({
                "entropy": entropy,
                "logp_expert_action": logp_expert.mean().item(),
                "action_margin": margin,
            })

            if hpGrid is not None:
                heavy = self.compute_heavy_metrics(pred_actions, expert_flat, hpGrid, num_filters)
                metrics.update(heavy)

        return metrics
    

class QPolicyBase(BCPolicyBase):
    """Adapts a network's output to a flat `(batch, num_actions)` Q-tensor.

    Implementations exist for each network output shape (flat scores,
    autoregressive heads, etc). The RL algorithm computes the TD loss; the
    adapter only handles the forward pass.
    """

    @abstractmethod
    def get_q_values(self, x_glob, x_bin) -> torch.Tensor:
        ...

    def select_action(self, x_glob, x_bin, action_mask=None) -> torch.Tensor:
        q_vals = self.get_q_values(x_glob, x_bin)
        if action_mask is not None:
            mask_val = torch.finfo(q_vals.dtype).min
            q_vals = q_vals.masked_fill(~action_mask.bool(), mask_val)
        return q_vals.argmax(dim=-1)


# --------------------------------------------------------------------------- #
# Shared metric helpers (free functions — usable from any policy/adapter)
# --------------------------------------------------------------------------- #

def compute_slew_distance(predicted_bins, expert_bins, hpGrid) -> float:
    predicted_coords = np.array((hpGrid.lon[predicted_bins], hpGrid.lat[predicted_bins]))
    expert_coords = np.array((hpGrid.lon[expert_bins], hpGrid.lat[expert_bins]))
    return float(geometry.angular_separation(predicted_coords, expert_coords).mean())


