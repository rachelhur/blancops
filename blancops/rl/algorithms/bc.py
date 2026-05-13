import torch

from blancops.rl.algorithms.base import AlgorithmBase
import logging
logger = logging.getLogger(__name__)

class BehaviorCloning(AlgorithmBase):
    def __init__(self, policy, optimizer, lr_scheduler, lr_scheduler_epoch_start=1, lr_scheduler_num_epochs=50, optimizer_kwargs=None, lr_scheduler_kwargs=None, device='cpu'):
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
         dones, action_masks, next_action_masks, bin_states, next_bin_states, slew_dists) = batch

        batch_dict = {
            'state': state.to(device=self.device, dtype=torch.float32),
            'bin_states': bin_states.to(device=self.device, dtype=torch.float32),
            'expert_actions': expert_actions_flat.view(-1).to(device=self.device, dtype=torch.long),
            'action_masks': action_masks.to(device=self.device, dtype=torch.bool),
            'slew_dists': slew_dists
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