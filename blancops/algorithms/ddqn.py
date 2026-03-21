import numpy as np
import torch
import torch.nn.functional as F

from blancops.core_rl.neural_nets import MLP, SingleScoreMLP, BinEmbeddingDQN, MultiScoreMLP
from blancops.math import geometry
from blancops.algorithms.base import AlgorithmBase
import logging
logger = logging.getLogger(__name__)

from pathlib import Path

class DDQN(AlgorithmBase):
    """
    Implementation of the DDQN algorithm. Uses AdamW optimizer and, optionally, a cosine annealing lr scheduler.

    Args
    ----
    obs_dim (int): size of each observation
    num_actions (int): number of total possible actions
    hidden_dim (int): hidden dimension size in DQN network
    gamma (float): 
    tau (float): 
    device (str): 
    lr (float): Learning rate
    loss_fxn (torch.nn.functional): Loss function (ie F.huber_loss, F.mse_loss)
    use_dqn (bool): 
    optimizer_kwargs (optional): 
    """

    def __init__(self, obs_dim, num_actions, hidden_dim, gamma, tau, device, lr, activation=None, target_update_freq=1000, loss_fxn=None, use_double=True, \
                 lr_scheduler='cosine_annealing', lr_scheduler_kwargs=None, optimizer_kwargs={},
                 lr_scheduler_epoch_start=1, lr_scheduler_num_epochs=5):
        super().__init__()

        assert loss_fxn is not None, "loss_fxn needs to be passed"
        self.loss_fxn = loss_fxn
        if use_double:
            self.name = 'DDQN'
        else:
            self.name = 'DQN'
        
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        self.policy_net = MLP(observation_dim=obs_dim, action_dim=num_actions, hidden_dim=hidden_dim, activation=activation).to(device)
        self.target_net = MLP(observation_dim=obs_dim, action_dim=num_actions, hidden_dim=hidden_dim, activation=activation).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.use_double = use_double
        self.target_update_freq = target_update_freq

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=False, **optimizer_kwargs)
        self.lr_scheduler = self._initialize_scheduler(lr_scheduler=lr_scheduler, lr_scheduler_kwargs=lr_scheduler_kwargs, optimizer=self.optimizer)

        if lr_scheduler is not None:
            self.lr_scheduler_epoch_start = lr_scheduler_epoch_start
            self.lr_scheduler_num_epochs = lr_scheduler_num_epochs

        assert loss_fxn is not None
        self.loss_fxn = loss_fxn
        self.val_metrics = ['val_loss', 'td_error', 'q_std', 'q_policy', 'q_expert', 'accuracy', 'total_grad_norm', 'ang_sep', 'unique_bins']
        
    def train_step(self, batch, epoch_num, step_num):
        state, actions, rewards, next_state, dones, action_masks = batch

        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1) # needs to be long for .gather()
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        next_state = torch.as_tensor(np.array(next_state), device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
        action_masks = torch.as_tensor(np.array(action_masks), device=self.device, dtype=torch.bool)
            
        # need to input (batch_size, obs_dim) into net - if obs_dim is 1, we get 1d tensor. Need to reshape
        if state.dim() == 1:
            state = state.unsqueeze(1)
            next_state = next_state.unsqueeze(1)
        
        # Get policy's q vals for current state
        q_vals = self.policy_net(state)
        q_current = q_vals.gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.use_double:
                # Select best action with policy net
                pol_q_next = self.policy_net(next_state)
                pol_q_next[~action_masks] = float('-inf') #-1e9
                pol_next_actions = pol_q_next.argmax(1).type(torch.long)

                # Evaluate policy's next actions using target net
                target_next_q = self.target_net(next_state)
                next_q_targ = target_next_q.gather(1, pol_next_actions.unsqueeze(1)).squeeze(1)

                td_target = rewards + self.gamma * (1 - dones) * next_q_targ
            else:
                next_q = self.target_net(next_state)
                # mask invalid actions
                next_q[~action_masks] = -1e9 # float('-inf')
                max_next_q = next_q.max(dim=1)[0]
                td_target = rewards + self.gamma * max_next_q * (1 - dones) # , dtype=torch.float32, device=device

        loss = self.loss_fxn(q_current, td_target)
        
        # optimize w/ backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        do_lr_scheduler_step = (self.lr_scheduler is not None
                                and epoch_num >= self.lr_scheduler_epoch_start
                                and epoch_num <= self.lr_scheduler_num_epochs + self.lr_scheduler_epoch_start
        )
        if do_lr_scheduler_step:
            self.lr_scheduler.step()

        if step_num % self.target_update_freq == 0:
            self._soft_update()

        # total_norm = torch.norm(torch.stack([
        #     p.grad.norm() for p in self.policy_net.parameters()
        #     if p.grad is not None
        # ]))


        return loss.item(), q_vals.mean().item()

    def _soft_update(self):
        # update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, obs, action_mask, epsilon=None):
        # if random sample less than epsilon, take random action
        if epsilon is not None:
            if np.random.random() < epsilon:
                valid_actions = np.where(action_mask)[0]
                action = np.random.choice(valid_actions)
                return int(action)

        # greedy selection from policy
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            q_values = self.policy_net(obs).squeeze(0)
            # mask invalid actions
            action_mask = torch.tensor(action_mask, device=self.device, dtype=torch.bool)
            q_values[~action_mask] = float('-inf')
            action = torch.argmax(q_values).item()
        return int(action)
    
    def val_step(self, eval_batch, hpGrid=None):
        state, actions, rewards, next_state, dones, action_masks = eval_batch

        with torch.no_grad():      
            # convert to tensors
            state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            actions = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1) # needs to be long for .gather()
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float32)
            dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
            action_masks = torch.tensor(np.array(action_masks), device=self.device, dtype=torch.bool)

            q_vals = self.policy_net(state)
            q_current = q_vals.gather(1, actions.unsqueeze(1) if actions.dim() == 1 else actions).squeeze()
            predicted_actions = q_vals.argmax(1)

            # Compute TD targets for loss
            if self.use_double:
                pol_next_q = self.policy_net(next_state)
                pol_next_q[~action_masks] = -1e9
                pol_next_actions = pol_next_q.argmax(1)
                
                target_next_q = self.target_net(next_state)
                next_q_vals = target_next_q.gather(1, pol_next_actions.unsqueeze(1)).squeeze()
            else:
                # DQN
                target_next_q = self.target_net(next_state).clone()
                target_next_q[~action_masks] = float('-inf')
                next_q_vals = target_next_q.max(1)[0]
            
            # Compute TD target: r + γ * Q(s', a') * (1 - done)
            td_target = rewards + self.gamma * next_q_vals * (1 - dones)
            
            # Compute TD error/loss
            loss = self.loss_fxn(q_current, td_target)

            # Compute metrics
            td_error_mean = (q_current - td_target).abs().mean()
            mean_accuracy = (predicted_actions == actions).float().mean()
            q_dataset_mean = q_vals.gather(1, actions.unsqueeze(1) if actions.dim() == 1 else actions).squeeze().mean()
            q_policy_mean = q_vals.max(dim=1)[0].mean()
            q_std = q_vals.std()

            if hpGrid is not None:
                # Get angular separation
                predicted_actions = predicted_actions.cpu()
                expert_actions = actions.cpu()
                predicted_coords = np.array((hpGrid.lon[predicted_actions], hpGrid.lat[predicted_actions]))
                expert_actions_coords = np.array((hpGrid.lon[expert_actions], hpGrid.lat[expert_actions]))
                ang_seps = geometry.angular_separation(predicted_coords, expert_actions_coords)
                ang_sep = ang_seps.mean()
                # Prediction diversity
                num_actions = len(hpGrid.lon)
                unique_preds = len(torch.unique(predicted_actions))
                unique_bins = unique_preds / num_actions
            else:
                ang_sep = 0

            return loss.item(), td_error_mean.item(), q_std.item(), q_policy_mean.item(), q_dataset_mean.item(), mean_accuracy.item(), ang_sep, unique_bins
            
