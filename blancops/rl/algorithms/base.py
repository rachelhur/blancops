import numpy as np
import torch
import torch.nn.functional as F

from blancops.rl.neural_nets.neural_nets import MLP, BinEmbeddingDQN, ContextualScoreMLP
from blancops.math import geometry

import logging
logger = logging.getLogger(__name__)

from pathlib import Path

class AlgorithmBase:
    """Computes losses, steps optimizer, and managers learning rates"""
    def __init__(self):
        super().__init__()
        
    def train_step(self, batch):
        raise NotImplementedError

    def select_action(self, state):
        raise NotImplementedError

    def save_checkpoint(self, filepath, epoch=None):
        """Saves everything needed to resume training."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            # Optional: Add optimizer state here later if you want to resume training
            # 'optimizer_state_dict': self.optimizer.state_dict(), 
            'epoch': epoch
        }
        torch.save(checkpoint, filepath)
    
    def export_for_deployment(self, filepath):
        """Saves purely the unwrapped policy weights for inference."""
        raw_state_dict = self.policy.state_dict()
        clean_state_dict = {}
        
        # Strip away any wrapper prefixes so the raw policy can load it natively
        for key, value in raw_state_dict.items():
            clean_key = key.replace("policy.", "", 1) if key.startswith("policy.") else key
            clean_state_dict[clean_key] = value
            
        torch.save(clean_state_dict, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])

    def _initialize_scheduler(self, lr_scheduler, lr_scheduler_kwargs, optimizer):
        if lr_scheduler is None:
            return None
        
        if lr_scheduler == 'cosine_annealing' or lr_scheduler == torch.optim.lr_scheduler.CosineAnnealingLR:
            assert lr_scheduler_kwargs is not None, "Cosine annealing lr scheduler requires T_max and eta_min kwargs"
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, **lr_scheduler_kwargs) 
        else:
            raise NotImplementedError
        return lr_scheduler