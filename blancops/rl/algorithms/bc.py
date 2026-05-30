import torch

from blancops.configs.enums import Algorithm
from blancops.rl.algorithms.base import AlgorithmBase
import logging
logger = logging.getLogger(__name__)

class BehaviorCloning(AlgorithmBase):
    name = Algorithm.BC
    def __init__(self, policy, optimizer, lr_scheduler, lr_scheduler_epoch_start=1, lr_scheduler_num_epochs=50, optimizer_kwargs=None, lr_scheduler_kwargs=None, device='cpu'):
        super().__init__(
            policy=policy,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_epoch_start=lr_scheduler_epoch_start,
            lr_scheduler_num_epochs=lr_scheduler_num_epochs,
            device=device,
        )
        
        # optimizer_kwargs = optimizer_kwargs or {}
        
    
    def _unpack_batch(self, batch):
        (state, expert_actions_flat, rewards, next_state,
         dones, action_masks, next_action_masks, bin_states, next_bin_states,
         slew_dists) = batch

        batch_dict = {
            'state': state.to(device=self.device, dtype=torch.float32),
            'bin_states': bin_states.to(device=self.device, dtype=torch.float32),
            'expert_actions': expert_actions_flat.view(-1).to(device=self.device, dtype=torch.long),
            'action_masks': action_masks.to(device=self.device, dtype=torch.bool),
            'slew_dists': slew_dists
        }
        return batch_dict
    
    def _compute_loss(self, batch_dict, hpGrid=None, compute_metrics=False):
        # BC delegates the entire loss/metrics computation to the policy,
        # because the loss strategy varies per policy (PureJoint, HybridMarginal, ...).
        return self.policy.compute_loss_and_metrics(batch_dict, hpGrid, compute_metrics)