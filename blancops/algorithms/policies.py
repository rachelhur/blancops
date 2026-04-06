"""
Wrapper class for policy networks
"""

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from abc import ABC, abstractmethod

from blancops.math import geometry

class PolicyBase(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def compute_loss_and_metrics(self, batch):
        pass
    
    @abstractmethod
    def select_action(self, x_glob, x_bin, action_mask=None):
        pass

    def _compute_standard_metrics(self, action_logits, expert_flat, action_masks, hpGrid=None):
        """
        Universally calculates entropy, margin, and physical metrics for any flat action policy.
        """
        metrics_dict = {}
        mask_value = torch.finfo(action_logits.dtype).min
        
        with torch.no_grad():
            # Clone and mask for accurate predictions
            masked_logits = action_logits.clone()
            masked_logits[~action_masks] = mask_value
            pred_actions = masked_logits.argmax(dim=1)
            
            # 1. Log Probabilities & Entropy
            logp = F.log_softmax(action_logits, dim=-1)
            logp_expert_actions = logp.gather(1, expert_flat.unsqueeze(1)).squeeze(1)
            
            p = F.softmax(action_logits, dim=-1)
            entropy = -(p * logp).sum(dim=-1).mean().item()

            # 2. Action Margin (Expert vs Next Best)
            _, num_actions = action_logits.shape
            z_expert = action_logits.gather(1, expert_flat.unsqueeze(1)).squeeze(1)
            expert_mask = F.one_hot(expert_flat, num_classes=num_actions).bool()
            z_max_other = action_logits.masked_fill(expert_mask, float("-inf")).max(dim=1).values
            margin = (z_expert - z_max_other).mean().item()

            # 3. Domain-Specific Heavy Metrics
            if hpGrid is not None:
                heavy_metrics_dict = self._compute_heavy_metrics(pred_actions, expert_flat, hpGrid)
                metrics_dict.update(heavy_metrics_dict)
            
            # Combine all universal metrics
            metrics_dict.update({
                'action_margin': margin,
                'entropy': entropy,
                'logp_expert_action': logp_expert_actions.mean().item(),
            })
            
        return metrics_dict
    
    def _compute_heavy_metrics(self, predicted_actions, expert_actions, hpGrid):
    # Default values for heavy metrics
        ang_sep = 0.0
        unique_bins = 0.0
        filter_accuracy = 0.0
        
        metrics = {}
        if hpGrid is not None:
            # Get angular separation
            predicted_actions = predicted_actions.cpu()
            expert_actions = expert_actions.cpu()
            metrics['accuracy'] = (predicted_actions == expert_actions).float().mean().item()

            if self.num_filters is not None and self.num_filters != 1:        
                # Get bins
                predicted_bins = predicted_actions // self.num_filters
                expert_bins = expert_actions // self.num_filters
                # Get filters
                predicted_filters = predicted_actions % self.num_filters
                expert_filters = expert_actions % self.num_filters
                # Calculate filter metrics
                filter_accuracy = (predicted_filters == expert_filters).float().mean().item()
                unique_filter_preds = len(torch.unique(predicted_filters))
                unique_filters = unique_filter_preds / self.num_filters if self.num_filters is not None else 0
                metrics['filter_accuracy'] = filter_accuracy
                metrics['unique_filters'] = unique_filters
            else:
                predicted_bins = predicted_actions
                expert_bins = expert_actions
                filter_accuracy = 0.

            bin_accuracy = (predicted_bins == expert_bins).float().mean().item()
            predicted_coords = np.array((hpGrid.lon[predicted_bins], hpGrid.lat[predicted_bins]))
            expert_actions_coords = np.array((hpGrid.lon[expert_bins], hpGrid.lat[expert_bins]))
            ang_seps = geometry.angular_separation(predicted_coords, expert_actions_coords)
            ang_sep = ang_seps.mean()

            # Prediction diversity
            num_actions = len(hpGrid.lon)
            unique_bin_preds = len(torch.unique(predicted_bins))
            unique_bins = unique_bin_preds / num_actions
            
            metrics.update({
                'ang_sep': ang_sep,
                'unique_bins': unique_bins,
                'bin_accuracy': bin_accuracy
            })
        return metrics

class PureJointPolicy(PolicyBase):
    def __init__(self, core_net, loss_function, num_filters):
        super().__init__()
        self.core_net = core_net
        self.loss_function = loss_function
        self.num_filters = num_filters
            
    def compute_loss_and_metrics(self, batch: dict, hpGrid=None, compute_metrics=False):
        action_logits = self.core_net(x_glob=batch['state'], x_bin=batch['bin_states'])
        action_masks = batch['action_masks']
        expert_flat = batch['expert_actions']
        
        # Apply mask
        mask_value = torch.finfo(action_logits.dtype).min
        action_logits = action_logits.masked_fill(~action_masks, mask_value)
        
        # Pure Joint Loss
        loss = self.loss_function(action_logits, expert_flat)
        
        # Metrics
        metrics_dict = {}
        if compute_metrics:
            metrics_dict = self._compute_standard_metrics(action_logits, expert_flat, action_masks, hpGrid)
            
        return loss, metrics_dict
    
    def select_action(self, x_glob, x_bin, action_mask=None):
        logits = self.core_net(x_glob=x_glob, x_bin=x_bin)
        if action_mask is not None:
            mask_val = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~action_mask, mask_val)
        return logits.argmax(dim=-1)

class PseudoAutoregressivePolicy(PolicyBase):
    def __init__(self, core_net, num_filters, filter_penalty=5.0):
        super().__init__()
        self.core_net = core_net
        self.num_filters = num_filters
        self.filter_penalty = filter_penalty
            
    def compute_loss_and_metrics(self, batch: dict, hpGrid=None, compute_metrics=False):
        action_logits = self.core_net(x_glob=batch['state'], x_bin=batch['bin_states'])
        action_masks = batch['action_masks']
        expert_flat = batch['expert_actions']
        
        # Apply mask
        mask_value = torch.finfo(action_logits.dtype).min
        action_logits = action_logits.masked_fill(~action_masks, mask_value)
        
        # 1. Reshape into [Batch, Bins, Filters]
        batch_size = action_logits.size(0)
        n_bins = action_logits.size(1) // self.num_filters
        logits_2d = action_logits.view(batch_size, n_bins, self.num_filters)
        
        # 2. Extract Targets
        expert_bins = expert_flat // self.num_filters
        expert_filters = expert_flat % self.num_filters
        
        # 3. Bin Loss (Marginalized over all filters)
        bin_logits = torch.logsumexp(logits_2d, dim=2) 
        bin_loss = F.cross_entropy(bin_logits, expert_bins)
        
        # 4. Filter Loss (Conditioned explicitly on the expert's bin)
        batch_idx = torch.arange(batch_size, device=action_logits.device)
        filter_logits_at_expert_bin = logits_2d[batch_idx, expert_bins, :]
        filter_loss = F.cross_entropy(filter_logits_at_expert_bin, expert_filters)
        
        # 5. Weighted Total Loss
        loss = bin_loss + (self.filter_penalty * filter_loss)
        
        # 6. Metrics
        metrics_dict = {}
        if compute_metrics:
            metrics_dict = self._compute_standard_metrics(action_logits, expert_flat, action_masks, hpGrid)
            # Add the specific sub-losses to track how the penalty affects training
            metrics_dict.update({
                'bin_loss': bin_loss.item(),
                'filter_loss': filter_loss.item()
            })
            
        return loss, metrics_dict
    
    def select_action(self, x_glob, x_bin, action_mask=None):
        logits = self.core_net(x_glob=x_glob, x_bin=x_bin)
        if action_mask is not None:
            mask_val = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~action_mask, mask_val)
        return logits.argmax(dim=-1)

class HybridMarginalPolicy(PolicyBase):
    def __init__(self, core_net, num_filters, bin_loss_function, filter_loss_function, joint_loss_function, alpha_bin=1.0, beta_filter=5.0, zeta_joint=0.1):
        # Remember: if using focal loss, increase beta (filter penalty weight). need to balance with the fractional multiplier of focal loss 
        super().__init__()
        self.core_net = core_net
        self.num_filters = num_filters
        
        # Loss weighting parameters
        self.alpha = alpha_bin # Bin weight
        self.beta = beta_filter   # Filter weight
        self.zeta = zeta_joint # Joint weight
                
        self.bin_loss_function = bin_loss_function
        self.filter_loss_function = filter_loss_function
        self.joint_loss_function = joint_loss_function
            
    def compute_loss_and_metrics(self, batch: dict, hpGrid=None, compute_metrics=False):
        action_logits = self.core_net(x_glob=batch['state'], x_bin=batch['bin_states'])
        action_masks = batch['action_masks']
        expert_flat = batch['expert_actions']
        
        # Apply mask
        mask_value = torch.finfo(action_logits.dtype).min
        action_logits = action_logits.masked_fill(~action_masks, mask_value)
        
        # 1. Reshape for Marginals
        batch_size = action_logits.size(0)
        n_bins = action_logits.size(1) // self.num_filters
        logits_2d = action_logits.view(batch_size, n_bins, self.num_filters)
        
        expert_bins = expert_flat // self.num_filters
        expert_filters = expert_flat % self.num_filters
        
        # 2. Calculate Marginal & Joint Losses
        bin_logits_marginal = torch.logsumexp(logits_2d, dim=2) 
        bin_loss = self.bin_loss_function(bin_logits_marginal, expert_bins)
        
        filter_logits_marginal = torch.logsumexp(logits_2d, dim=1) 
        filter_loss = self.filter_loss_function(filter_logits_marginal, expert_filters)

        joint_loss = self.joint_loss_function(action_logits, expert_flat)
        
        # 3. Weighted Total Loss
        total_loss = (self.alpha * bin_loss) + (self.beta * filter_loss) + (self.zeta * joint_loss)
        
        # 4. Metrics
        metrics_dict = {}
        if compute_metrics:
            metrics_dict = self._compute_standard_metrics(action_logits, expert_flat, action_masks, hpGrid)
            # Add specific sub-losses to metrics so you can track them in TensorBoard/WandB
            metrics_dict.update({
                'bin_loss': bin_loss.item(),
                'filter_loss': filter_loss.item(),
                'joint_loss': joint_loss.item()
            })
            
        return total_loss, metrics_dict
    
    def select_action(self, x_glob, x_bin, action_mask=None):
        logits = self.core_net(x_glob=x_glob, x_bin=x_bin)
        if action_mask is not None:
            mask_val = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~action_mask, mask_val)
        return logits.argmax(dim=-1)
    
class AutoregressiveActionPolicy(PolicyBase):
    def __init__(self, core_net, num_filters, loss_function=None):
        super().__init__()
        self.core_net = core_net
        self.num_filters = num_filters
        self._filt_idx = self.core_net._filt_idx
        self._bin_idx = self.core_net._bin_idx
        
    def compute_loss_and_metrics(self, batch, hpGrid=None, compute_metrics=False):
        x_glob = batch['state']
        x_bin = batch['bin_states']
        expert_actions_flat = batch['expert_actions']
        action_masks = batch['action_masks']
        
        expert_bins = expert_actions_flat // self.num_filters
        expert_filters = expert_actions_flat % self.num_filters
        if self._filt_idx == 0: 
            expert_actions_multidim = torch.stack([expert_filters, expert_bins], dim=1)
        else: 
            expert_actions_multidim = torch.stack([expert_bins, expert_filters], dim=1)
        
        _, joint_logp, joint_entropy = self.core_net(
            x_glob=x_glob, 
            x_bin=x_bin, 
            action=expert_actions_multidim, 
            action_mask=action_masks
        )
        
        loss = -joint_logp.mean()
        
        metrics_dict = {}
        if compute_metrics:
            with torch.no_grad():
                metrics_dict['entropy'] = joint_entropy.mean().item()
                metrics_dict['logp_expert_action'] = joint_logp.mean().item()
                
                predicted_actions_multidim, _, _ = self.core_net(x_glob=x_glob, x_bin=x_bin, action_mask=action_masks, action=None)
                
                pred_filters = predicted_actions_multidim[:, self._filt_idx] # if filter first
                pred_bins = predicted_actions_multidim[:, self._bin_idx]
                predicted_actions_flat = pred_bins * self.num_filters + pred_filters
                
                if hpGrid is not None:
                    heavy_metrics_dict = self._compute_heavy_metrics(predicted_actions_flat, expert_actions_flat, hpGrid)
                    metrics_dict.update(heavy_metrics_dict)
            
        return loss, metrics_dict
    
    def select_action(self, x_glob, x_bin, action_mask=None):
        sampled_actions, _, _ = self.core_net(x_glob=x_glob, x_bin=x_bin, action_mask=action_mask, action=None)
        
        pred_filters = sampled_actions[:, self._filt_idx] # if filter first
        pred_bins = sampled_actions[:, self._bin_idx]
        best_actions_flat = pred_bins * self.num_filters + pred_filters
        return best_actions_flat
    
class QNetPolicyBase(PolicyBase):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_q_values(self, state, bin_states):
        pass
    
    def compute_loss_and_metrics(self, batch, hpGrid=None, compute_metrics=False):
        raise NotImplementedError("RL algorithm handles TD loss for Q-networks.")

    def select_action(self, x_glob, x_bin, action_mask=None):
        q_vals = self.get_q_values(state=x_glob, bin_states=x_bin)
        
        if action_mask is not None:
            mask_val = torch.finfo(q_vals.dtype).min
            q_vals = q_vals.masked_fill(~action_mask.bool(), mask_val)
            
        return q_vals.argmax(dim=-1)

class FlatQNetWrapper(QNetPolicyBase):
    def __init__(self, core_net):
        super().__init__()
        self.core_net = core_net

    def get_q_values(self, x_glob, x_bin):
        return self.core_net(x_glob=x_glob, x_bin=x_bin)

class AutoregressiveQNetWrapper(QNetPolicyBase):
    def __init__(self, core_ar_net, num_filters):
        super().__init__()
        self.core_net = core_ar_net
        self.num_filters = num_filters
        self._filt_idx = self.core_net._filt_idx
        self._bin_idx = self.core_net._bin_idx
        
    def get_q_values(self, x_glob, x_bin):

        # 1. GET LATENT STATE        
        x_latent = self.core_net.state_encoder(x_glob, x_bin)
        batch_size = x_latent.size(0)
        
        # 2. GET Q-VALUES FOR FIRST HEAD
        q_1 = self.core_net.action_heads[0](x_latent)
        dim1 = q_1.size(1) # First head's action size (ie nfilters if filter first)

        # 3. GET FULL Q-TABLE FOR SECOND HEAD (conditioned on all first head choices)
        all_choices_1 = torch.arange(dim1, device=x_latent.device)
        emb_1 = self.core_net.action_embeddings[0](all_choices_1) # Shape: [dim1, EmbDim]

        # Broadcast latent state and embeddings for batch processing
        x_latent_exp = x_latent.unsqueeze(1).expand(-1, dim1, -1) # (batch_size, dim1, latent_dim)
        emb_1_exp = emb_1.unsqueeze(0).expand(batch_size, -1, -1) # (batch_size, dim1, emb_dim)

        x_current_2 = torch.cat([x_latent_exp, emb_1_exp], dim=-1)
        
        # Shape: (batch_size, dim1, dim2)
        q_2 = self.core_net.action_heads[1](x_current_2)

        # Q(s, a1, a2) = Q(s, a1) + Q(s, a2 | a1)
        q_joint = q_1.unsqueeze(2) + q_2 # Shape: [Batch, Dim1, Dim2]

        # Map (dim1, dim2) to flat index (flat_action = (bin * nfilters) + filter)
        if self._filt_idx == 0:
            # Transpose to (batch, bins, filters) to match dataloader's expected format
            q_joint = q_joint.transpose(1, 2)
        flat_joint_q_values = q_joint.flatten(start_dim=1)

        return flat_joint_q_values
    
from blancops.data_processing.constants import FILTER_ALPHA_WEIGHTS
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', use_alpha=True):
        """
        Modulates Cross Entropy loss for imbalanced classes with modulating factor gamma
        From https://arxiv.org/abs/1708.02002
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.use_alpha = use_alpha
        self.alpha = torch.tensor(FILTER_ALPHA_WEIGHTS, dtype=torch.float32)

    def forward(self, logits, targets):
        # Get loss = - log(p_t) ; ie, loss for target class
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Get prob p_t = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)

        # Apply focal scaling factor: (1 - p_t)^gamma
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        if self.use_alpha:
            self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss