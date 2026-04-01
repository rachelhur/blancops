import numpy as np
import torch
import torch.nn.functional as F

from blancops.core_rl.neural_nets import MLP, AutoregressiveDiscreteNet, MultiHeadMultiScoreNet, SingleScoreMLP, BinEmbeddingDQN, ScoreMLP
from blancops.math import geometry
from blancops.algorithms.base import AlgorithmBase
from blancops.data_processing.constants import GRID_NETWORKS
import logging
logger = logging.getLogger(__name__)

from pathlib import Path

class BehaviorCloning(AlgorithmBase):
    def __init__(self, policy, optimizer, lr_scheduler=None, lr_scheduler_epoch_start=1, lr_scheduler_num_epochs=50, optimizer_kwargs=None, lr_scheduler_kwargs=None, device='cpu'):
        super().__init__()
        self.name = 'BC'
        self.device = device
        self.device_type_str = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        self.amp_dtype = torch.bfloat16
            
        self.policy = policy.to(self.device)
        optimizer_kwargs = optimizer_kwargs or {}
        
        self.optimizer = optimizer
        self.lr_scheduler = self._initialize_scheduler(lr_scheduler, lr_scheduler_kwargs, self.optimizer)
        if lr_scheduler is not None:
            logger.debug(f'lr_scheduler is {self.lr_scheduler}')
            self.lr_scheduler_epoch_start = lr_scheduler_epoch_start
            self.lr_scheduler_num_epochs = lr_scheduler_num_epochs
    
    def _unpack_batch(self, batch):
        (state, expert_actions_flat, rewards, next_state, 
         dones, action_masks, next_action_masks, bin_states, next_bin_states) = batch

        batch_dict = {
            'state': state.to(device=self.device, dtype=torch.float32),
            'bin_states': bin_states.to(device=self.device, dtype=torch.float32),
            'expert_actions': expert_actions_flat.view(-1).to(device=self.device, dtype=torch.long),
            'action_masks': action_masks.to(device=self.device, dtype=torch.bool)
        }
        return batch_dict
    
    def train_step(self, batch, epoch_num, step_num=None, hpGrid=None, compute_metrics=False):
        """
        Train the policy to mimic expert actions from offline data
        """
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        batch_dict = self._unpack_batch(batch)
        
        # Compute loss
        with torch.amp.autocast(self.device_type_str, dtype=self.amp_dtype):
            loss, metrics_dict = self.policy.compute_loss_and_metrics(batch_dict, hpGrid, compute_metrics)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        do_lr_scheduler_step = (self.lr_scheduler is not None
                                and epoch_num >= self.lr_scheduler_epoch_start
                                and epoch_num <= self.lr_scheduler_num_epochs + self.lr_scheduler_epoch_start
        )
        if do_lr_scheduler_step:
            self.lr_scheduler.step()
            
        metrics_dict['train_loss'] = loss.item()
        return metrics_dict
    
    def val_step(self, batch, hpGrid=None):
        """Evaluates the policy without updating weights."""
        self.policy.eval()
        
        batch_dict = self._unpack_batch(batch)
        
        with torch.no_grad():
            loss, metrics_dict = self.policy.compute_loss_and_metrics(batch_dict, hpGrid, compute_metrics=True)

        metrics_dict['val_loss'] = loss.item()
        return metrics_dict

    def select_action(self, x_glob, x_bin, action_mask, epsilon=None):
        self.policy.eval()
        with torch.no_grad():
            x_glob = torch.as_tensor(x_glob, dtype=torch.float32, device=self.device)
            x_bin = torch.as_tensor(x_bin, dtype=torch.float32, device=self.device)
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            
            # Handle unbatched environment steps
            if x_glob.dim() == 1:
                x_glob = x_glob.unsqueeze(0)
            if x_bin.dim() == 2:
                x_bin = x_bin.unsqueeze(0)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            
            with torch.amp.autocast(device_type=self.device_type_str, dtype=self.amp_dtype):
                action_tensor = self.policy.get_action(x_glob, x_bin, action_mask)

            return int(action_tensor.item())