import numpy as np
import torch

def compute_airmass_reward(airmass):
    """
    Compute the airmass reward based on the given airmass value.

    Parameters:
    airmass (np.ndarray | torch.Tensor): A tensor containing the airmass values.

    Returns:
    torch.Tensor: A tensor containing the computed airmass rewards.
    """
    # Ensure that the airmass values are non-negative
    airmass = torch.clamp(airmass, min=0.0)

    # Compute the reward as the negative of the airmass
    reward = -airmass

    return reward

def compute_slew_time_reard(slew_time):
    """
    Compute the slew time reward based on the given slew time value.

    Parameters:
    slew_time (np.ndarray | torch.Tensor): A tensor containing the slew time values.

    Returns:
    reward: (np.ndarray | torch.Tensor): Returns an array or tensor (depending on input type) containing the computed slew time rewards.
    """
    backend = torch if isinstance(slew_time, torch.Tensor) else np
    # Ensure that the slew time values are non-negative
    if backend == torch:
        slew_time = torch.clamp(slew_time, min=0.0)
    else:
        slew_time = np.clip(slew_time, a_min=0.0, a_max=None)

    # Compute the reward as the negative of the slew time
    reward = -slew_time

    return reward