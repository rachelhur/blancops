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
                metrics['accuracy'] = (predicted_actions == expert_actions).float().mean().item()
                
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

class FlatActionPolicy(PolicyBase):
    def __init__(self, core_net, loss_function, num_filters):
        super().__init__()
        self.core_net = core_net
        self.loss_function = loss_function
        self.num_filters = num_filters
            
    def compute_loss_and_metrics(self, batch: dict, hpGrid=None, compute_metrics=False):
        x_glob = batch['state']
        x_bin = batch['bin_states']
        expert_bins = batch['expert_actions']
        action_masks = batch['action_masks']
        
        action_logits = self.core_net(x_glob=x_glob, x_bin=x_bin)
        mask_value = torch.finfo(action_logits.dtype).min

        action_logits = action_logits.masked_fill(~action_masks, mask_value)
        loss = self.loss_function(action_logits, expert_bins)
        
        metrics_dict = {}
        if compute_metrics:
            with torch.no_grad():
                masked_logits = action_logits.clone()
                masked_logits[~action_masks] = mask_value
                pred_bins = masked_logits.argmax(dim=1)
                
                # Log Probabilities & Entropy
                logp = F.log_softmax(action_logits, dim=-1)
                logp_expert_actions = logp.gather(1, expert_bins.unsqueeze(1)).squeeze(1)
                
                p = F.softmax(action_logits, dim=-1)
                entropy = -(p * logp).sum(dim=-1).mean().item()

                # Margin (Expert vs Next Best)
                _, num_actions = action_logits.shape
                z_expert = action_logits.gather(1, expert_bins.unsqueeze(1)).squeeze(1)
                expert_mask = F.one_hot(expert_bins, num_classes=num_actions).bool()
                z_max_other = action_logits.masked_fill(expert_mask, float("-inf")).max(dim=1).values
                margin = (z_expert - z_max_other).mean().item()

                if hpGrid is not None:
                    heavy_metrics_dict = self._compute_heavy_metrics(pred_bins, expert_bins, hpGrid)
                
                metrics_dict.update(heavy_metrics_dict)
                metrics_dict.update({
                    'action_margin': margin,
                    'entropy': entropy,
                    'logp_expert_action': logp_expert_actions.mean().item(),
                })
        
        return loss, metrics_dict
    
    def select_action(self, x_glob, x_bin, action_mask=None):
        logits = self.core_net(x_glob=x_glob, x_bin=x_bin)
        mask_val = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~action_mask, mask_val)
        best_actions = logits.argmax(dim=-1)
        return best_actions

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