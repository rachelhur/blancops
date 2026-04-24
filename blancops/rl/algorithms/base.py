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

    def _initialize_scheduler(self, lr_scheduler, lr_scheduler_kwargs, optimizer):
        if lr_scheduler is None:
            return None
        
        if lr_scheduler == 'cosine_annealing' or lr_scheduler == torch.optim.lr_scheduler.CosineAnnealingLR:
            assert lr_scheduler_kwargs is not None, "Cosine annealing lr scheduler requires T_max and eta_min kwargs"
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, **lr_scheduler_kwargs) 
        else:
            raise NotImplementedError
        return lr_scheduler