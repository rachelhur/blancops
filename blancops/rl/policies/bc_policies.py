"""BC loss strategies. Each defines how expert actions map to a loss given
the network's output shape."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from blancops.rl.policies.base import (
    BCPolicyBase,
)
from blancops.rl.policies.loss_function import SlewDistanceFocalLoss


class BCPureJointPolicy(BCPolicyBase):
    """Flat cross-entropy over the full joint (bin × filter) action space."""

    def __init__(self, core_net: nn.Module, loss_function: nn.Module, num_filters: int):
        super().__init__()
        self.core_net = core_net
        self.loss_function = loss_function
        self.num_filters = num_filters

    def compute_loss_and_metrics(self, batch, hpGrid=None, compute_metrics=False):
        action_logits = self.core_net(x_glob=batch["state"], x_bin=batch["bin_states"])
        action_masks = batch["action_masks"]
        expert_flat = batch["expert_actions"]
        exp_slew_dists = batch.get("slew_dists", None)

        mask_val = torch.finfo(action_logits.dtype).min
        action_logits = action_logits.masked_fill(~action_masks, mask_val)

        # Dispatch on loss-function signature.
        if exp_slew_dists is not None and isinstance(self.loss_function, SlewDistanceFocalLoss):
            loss = self.loss_function(action_logits, expert_flat, exp_slew_dists)
        else:
            loss = self.loss_function(action_logits, expert_flat)

        metrics: dict = {}
        if compute_metrics:
            metrics = self.compute_standard_metrics(
                action_logits, expert_flat, action_masks, self.num_filters, hpGrid
            )
            metrics.update(self._marginal_loss_diagnostics(action_logits, expert_flat))

        return loss, metrics

    def _marginal_loss_diagnostics(self, action_logits, expert_flat) -> dict:
        """Auxiliary: report what bin- and filter-marginal losses would be
        if you trained them separately. Useful for understanding what the
        joint loss is implicitly weighting."""
        with torch.no_grad():
            batch_size = action_logits.size(0)
            n_bins = action_logits.size(1) // self.num_filters
            logits_2d = action_logits.view(batch_size, n_bins, self.num_filters)

            log_norm = torch.logsumexp(action_logits, dim=-1, keepdim=True)   # (batch, 1)

            bin_log_probs    = torch.logsumexp(logits_2d, dim=2) - log_norm   # (batch, n_bins)
            filter_log_probs = torch.logsumexp(logits_2d, dim=1) - log_norm   # (batch, n_filters)

            expert_bin = expert_flat // self.num_filters
            expert_filter = expert_flat % self.num_filters

            bin_loss = F.nll_loss(bin_log_probs, expert_bin)
            filter_loss = F.nll_loss(filter_log_probs, expert_filter)

            # batch_size = action_logits.size(0)
            # n_bins = action_logits.size(1) // self.num_filters
            # probs = F.softmax(
            #     action_logits.view(batch_size, n_bins, self.num_filters), dim=-1
            # )
            # bin_probs = probs.sum(dim=-1)
            # filter_probs = probs.sum(dim=-2)

            # expert_bin = expert_flat // self.num_filters
            # expert_filter = expert_flat % self.num_filters

            # bin_loss = F.nll_loss(torch.log(bin_probs + 1e-9), expert_bin)
            # filter_loss = F.nll_loss(torch.log(filter_probs + 1e-9), expert_filter)
        return {"bin_loss": bin_loss.item(), "filter_loss": filter_loss.item()}

    def select_action(self, x_glob, x_bin, action_mask=None):
        logits = self.core_net(x_glob=x_glob, x_bin=x_bin)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return logits.argmax(dim=-1)


class BCPseudoAutoregressivePolicy(BCPolicyBase):
    """Factored loss on simultaneous logits: marginalized bin loss + filter
    loss conditioned on the expert bin. Encourages the network to get the
    bin right first, then the filter given that bin."""

    def __init__(self, core_net: nn.Module, num_filters: int, filter_penalty: float = 5.0):
        super().__init__()
        self.core_net = core_net
        self.num_filters = num_filters
        self.filter_penalty = filter_penalty

    def compute_loss_and_metrics(self, batch, hpGrid=None, compute_metrics=False):
        action_logits = self.core_net(x_glob=batch["state"], x_bin=batch["bin_states"])
        action_masks = batch["action_masks"]
        expert_flat = batch["expert_actions"]

        mask_val = torch.finfo(action_logits.dtype).min
        action_logits = action_logits.masked_fill(~action_masks, mask_val)

        batch_size = action_logits.size(0)
        n_bins = action_logits.size(1) // self.num_filters
        logits_2d = action_logits.view(batch_size, n_bins, self.num_filters)

        expert_bins = expert_flat // self.num_filters
        expert_filters = expert_flat % self.num_filters

        bin_logits = torch.logsumexp(logits_2d, dim=2)
        bin_loss = F.cross_entropy(bin_logits, expert_bins)

        batch_idx = torch.arange(batch_size, device=action_logits.device)
        filter_logits_at_expert_bin = logits_2d[batch_idx, expert_bins, :]
        filter_loss = F.cross_entropy(filter_logits_at_expert_bin, expert_filters)

        loss = bin_loss + self.filter_penalty * filter_loss

        metrics: dict = {}
        if compute_metrics:
            metrics = self.compute_standard_metrics(
                action_logits, expert_flat, action_masks, self.num_filters, hpGrid
            )
            metrics["bin_loss"] = bin_loss.item()
            metrics["filter_loss"] = filter_loss.item()

        return loss, metrics

    def select_action(self, x_glob, x_bin, action_mask=None):
        logits = self.core_net(x_glob=x_glob, x_bin=x_bin)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return logits.argmax(dim=-1)



class BCHybridMarginalPolicy(BCPolicyBase):
    """Weighted sum of bin-marginal + filter-marginal + joint losses.

    α·bin_loss + β·filter_loss + ζ·joint_loss. If using focal loss on the
    filter head, bump β to balance against the focal-loss scaling.
    """

    def __init__(
        self,
        core_net: nn.Module,
        num_filters: int,
        bin_loss_function: nn.Module,
        filter_loss_function: nn.Module,
        joint_loss_function: nn.Module,
        alpha_bin: float = 1.0,
        beta_filter: float = 5.0,
        zeta_joint: float = 0.1,
    ):
        super().__init__()
        self.core_net = core_net
        self.num_filters = num_filters
        self.alpha = alpha_bin
        self.beta = beta_filter
        self.zeta = zeta_joint
        self.bin_loss_function = bin_loss_function
        self.filter_loss_function = filter_loss_function
        self.joint_loss_function = joint_loss_function

    def compute_loss_and_metrics(self, batch, hpGrid=None, compute_metrics=False):
        action_logits = self.core_net(x_glob=batch["state"], x_bin=batch["bin_states"])
        action_masks = batch["action_masks"]
        expert_flat = batch["expert_actions"]

        mask_val = torch.finfo(action_logits.dtype).min
        action_logits = action_logits.masked_fill(~action_masks, mask_val)

        batch_size = action_logits.size(0)
        n_bins = action_logits.size(1) // self.num_filters
        logits_2d = action_logits.view(batch_size, n_bins, self.num_filters)

        expert_bins = expert_flat // self.num_filters
        expert_filters = expert_flat % self.num_filters

        bin_logits_marginal = torch.logsumexp(logits_2d, dim=2)
        bin_loss = self.bin_loss_function(bin_logits_marginal, expert_bins)

        filter_logits_marginal = torch.logsumexp(logits_2d, dim=1)
        filter_loss = self.filter_loss_function(filter_logits_marginal, expert_filters)

        joint_loss = self.joint_loss_function(action_logits, expert_flat)

        total_loss = self.alpha * bin_loss + self.beta * filter_loss + self.zeta * joint_loss

        metrics: dict = {}
        if compute_metrics:
            metrics = self.compute_standard_metrics(
                action_logits, expert_flat, action_masks, self.num_filters, hpGrid
            )
            metrics.update({
                "bin_loss": bin_loss.item(),
                "filter_loss": filter_loss.item(),
                "joint_loss": joint_loss.item(),
            })

        return total_loss, metrics

    def select_action(self, x_glob, x_bin, action_mask=None):
        logits = self.core_net(x_glob=x_glob, x_bin=x_bin)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return logits.argmax(dim=-1)


class BCAutoregressivePolicy(BCPolicyBase):
    """Pairs with `AutoregressiveNet`. The network samples actions
    sequentially with embedding conditioning; the loss is the negative
    joint log-probability of the expert action sequence."""

    def __init__(self, core_net: nn.Module, num_filters: int):
        super().__init__()
        self.core_net = core_net
        self.num_filters = num_filters
        self._filt_idx = core_net._filt_idx
        self._bin_idx = core_net._bin_idx

    def compute_loss_and_metrics(self, batch, hpGrid=None, compute_metrics=False):
        x_glob = batch["state"]
        x_bin = batch["bin_states"]
        expert_flat = batch["expert_actions"]
        action_masks = batch["action_masks"]

        expert_bins = expert_flat // self.num_filters
        expert_filters = expert_flat % self.num_filters
        if self._filt_idx == 0:
            expert_multidim = torch.stack([expert_filters, expert_bins], dim=1)
        else:
            expert_multidim = torch.stack([expert_bins, expert_filters], dim=1)

        _, joint_logp, joint_entropy = self.core_net(
            x_glob=x_glob,
            x_bin=x_bin,
            action=expert_multidim,
            action_mask=action_masks,
        )

        loss = -joint_logp.mean()

        metrics: dict = {}
        if compute_metrics:
            with torch.no_grad():
                metrics["entropy"] = joint_entropy.mean().item()
                metrics["logp_expert_action"] = joint_logp.mean().item()

                pred_multidim, _, _ = self.core_net(
                    x_glob=x_glob, x_bin=x_bin, action_mask=action_masks, action=None
                )
                pred_filters = pred_multidim[:, self._filt_idx]
                pred_bins = pred_multidim[:, self._bin_idx]
                pred_flat = pred_bins * self.num_filters + pred_filters

                if hpGrid is not None:
                    heavy = self.compute_heavy_metrics(pred_flat, expert_flat, hpGrid, self.num_filters)
                    metrics.update(heavy)

        return loss, metrics

    def select_action(self, x_glob, x_bin, action_mask=None):
        sampled, _, _ = self.core_net(
            x_glob=x_glob, x_bin=x_bin, action_mask=action_mask, action=None
        )
        pred_filters = sampled[:, self._filt_idx]
        pred_bins = sampled[:, self._bin_idx]
        return pred_bins * self.num_filters + pred_filters
    