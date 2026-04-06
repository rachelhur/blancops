import numpy as np
import torch
import torch.nn.functional as F

from blancops.core_rl.neural_nets import MLP, BinEmbeddingDQN, ScoreMLP
from blancops.math import geometry

import logging
logger = logging.getLogger(__name__)

from pathlib import Path

class AlgorithmBase:
    def __init__(self):
        super().__init__()
        
    def train_step(self, batch):
        raise NotImplementedError

    def select_action(self, state):
        raise NotImplementedError

    def save(self, filepath):
        torch.save({'policy': self.policy.state_dict()}, filepath)
    
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