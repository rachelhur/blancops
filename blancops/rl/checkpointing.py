import random

import numpy as np
import torch
from pathlib import Path

import os
import json
import torch
import logging

logger = logging.getLogger(__name__)

class Checkpointer:
    def __init__(self, outdir: Path, top_k: int = 1, mode: str = 'min'):
        """
        Args:
            outdir: Directory to save checkpoints.
            top_k: Maximum number of best models to keep on disk.
            mode: 'min' if lower metric is better (e.g., loss, ang_sep), 'max' for accuracy/reward.
        """
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        
        self.top_k = top_k
        self.mode = mode
        
        # Tracks our best models: list of dicts [{'filepath': str, 'metric': float}]
        self.best_checkpoints = [] 
        self._history_file = self.outdir / "checkpoint_history.json"
        self._load_history()

    def _load_history(self):
        """Loads previous checkpoint history if resuming a run."""
        if self._history_file.exists():
            try:
                with open(self._history_file, 'r') as f:
                    self.best_checkpoints = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Could not read checkpoint history. Starting fresh.")

    def _save_history(self):
        with open(self._history_file, 'w') as f:
            json.dump(self.best_checkpoints, f)

    def save_training_state(self, algorithm, epoch: int, metric_value: float, is_best: bool = False, norm_stats: dict = None):
        """Saves the state and manages the Top-K logic."""
        
        # Create the checkpoint dictionary
        checkpoint = {
            'policy_state_dict': algorithm.policy.state_dict(),
            'optimizer_state_dict': algorithm.optimizer.state_dict(),
            'epoch': epoch,
            'metric': metric_value,
            'norm_stats': norm_stats,
            'rng_states': {
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
                'python': random.getstate()
            }
        }
        
        # 1. ALWAYS overwrite the 'latest' file for easy resuming
        torch.save(checkpoint, self.outdir / 'latest_checkpoint.pt')

        # 2. Process Top-K if it's a new best
        if is_best:
            checkpoint_name = self.outdir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_name)
            
            # Add to tracker
            self.best_checkpoints.append({
                'filepath': str(checkpoint_name), 
                'metric': float(metric_value)
            })
            
            # Sort the list (best models at the beginning, worst at the end)
            reverse_sort = True if self.mode == 'max' else False
            self.best_checkpoints.sort(key=lambda x: x['metric'], reverse=reverse_sort)

            # 3. Prune the worst model if we exceeded our Top K limit
            if len(self.best_checkpoints) > self.top_k:
                worst_checkpoint = self.best_checkpoints.pop(-1)
                worst_path = Path(worst_checkpoint['filepath'])
                
                if worst_path.exists():
                    os.remove(worst_path)
                    logger.debug(f"Pruned older checkpoint: {worst_path.name}")
                    
            self._save_history()

    def export_deployment_model(self, policy, norm_stats: dict, filename="model.pt"):
        """Saves a stripped-down, prefix-free state dict for deployment."""
        raw_state_dict = policy.state_dict()
        clean_state_dict = {}
        
        for key, value in raw_state_dict.items():
            clean_key = key.replace("policy.", "", 1) if key.startswith("policy.") else key
            clean_state_dict[clean_key] = value
        
        deployment_package = {
            'model_state_dict': clean_state_dict,
            'norm_stats': norm_stats
        }
        torch.save(deployment_package, self.outdir / filename)
        logger.info(f"Deployment model and norm stats saved to {filename}")