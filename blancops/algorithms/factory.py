import logging
logger = logging.getLogger(__name__)


import torch.nn as nn
import torch.nn.functional as F

from blancops.algorithms.ddqn import DDQN
from blancops.algorithms.bc import BehaviorCloning

def setup_algorithm(algorithm_name=None, num_actions=None, loss_fxn=None, hidden_dim=None, lr=None, lr_scheduler=None, device=None, 
                    lr_scheduler_kwargs=None, gamma=None, tau=None, lr_scheduler_epoch_start=None, lr_scheduler_num_epochs=None, activation=None, 
                    grid_network=None, n_global_features=None, n_bin_features=0, num_filters=None, embedding_dim=None, use_contextual_gating=False):
    assert loss_fxn is not None

    # Initialize activation functions
    if type(activation) == str:
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'mish':
            activation = nn.Mish
        elif activation == 'swish':
            activation = nn.SiLU
        else:
            raise NotImplementedError(f"Activation function '{activation}' not implemented.")

    # Initialize activation functions
    if type(loss_fxn) == str:
        if loss_fxn == 'mse':
            loss_fxn = nn.MSELoss(reduction='mean')
        elif loss_fxn == 'huber':
            loss_fxn = nn.HuberLoss(reduction='mean')
        elif loss_fxn == 'cross_entropy':
            loss_fxn = nn.CrossEntropyLoss(reduction='mean')
        else:
            raise NotImplementedError(f"{loss_fxn} not yet implemented.")
        
    # Set up model hyperparameters that are algorithm independent
    model_hyperparams = {
        'num_actions': num_actions, 
        'hidden_dim': hidden_dim,
        'lr': lr,
        'lr_scheduler': lr_scheduler,
        'lr_scheduler_kwargs': lr_scheduler_kwargs,
        'lr_scheduler_epoch_start': lr_scheduler_epoch_start,
        'lr_scheduler_num_epochs': lr_scheduler_num_epochs,
        'loss_fxn': loss_fxn,
        'activation': activation,
        'grid_network': grid_network,
        'n_global_features': n_global_features,
        'n_bin_features': n_bin_features,
        'embedding_dim': embedding_dim,
        'num_filters': num_filters,
        'use_contextual_gating': use_contextual_gating
    }
        
    if algorithm_name == 'DDQN' or algorithm_name == 'DQN':
        assert gamma is not None, "Gamma (discount factor) must be specified for DDQN."
        assert tau is not None, "Tau (target network update rate) must be specified for DDQN."
        # assert loss_fxn in ['mse', 'huber'], "DDQN only supports mse or huber loss functions."
        
        if loss_fxn is not None and type(loss_fxn) != str:
            loss_fxn = loss_fxn

        else:
            raise NotImplementedError(f'Loss function {loss_fxn} not yet implemented for {algorithm_name}')

        model_hyperparams.update( {
            'gamma': gamma,
            'tau': tau,
            'use_double': algorithm_name == 'ddqn',
            'loss_fxn': loss_fxn
        } )

        algorithm = DDQN(
            device=device,
            **model_hyperparams
        )

    elif algorithm_name == 'BC':
        if loss_fxn is not None and type(loss_fxn) != str:
            loss_fxn = loss_fxn
        else:
            raise NotImplementedError(f'Loss function {loss_fxn} not yet implemented for {algorithm_name}')
        
        model_hyperparams.update({
        'loss_fxn': loss_fxn
        })
        algorithm = BehaviorCloning(
            device=device,
            **model_hyperparams
        )
    else:
        raise NotImplementedError(f"{algorithm_name} not yet implemented")
    return algorithm

