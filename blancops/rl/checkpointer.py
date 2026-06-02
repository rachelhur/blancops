import random

import numpy as np
import torch
from pathlib import Path

import os
import json
import torch
import logging

logger = logging.getLogger(__name__)


def _serialize_numpy_rng_state(state):
    # np.random.get_state() returns ('MT19937', np.array([...], uint32), pos, has_gauss, cached_gaussian)
    # Storing the numpy array directly breaks weights_only=True; convert to list.
    return (state[0], state[1].tolist(), state[2], state[3], float(state[4]))


class Checkpointer:
    def __init__(self, outdir: Path, top_k: int = 1, mode: str = 'min', overwrite=False, hard_overwrite=False):
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

        if hard_overwrite:
            self._hard_reset()
        elif overwrite:
            self._soft_reset()
        else:
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


    def _is_better(self, a: float, b: float) -> bool:
        """Return True if metric a is better than metric b."""
        if self.mode == "min":
            return a < b
        elif self.mode == "max":
            return a > b
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


    def save_training_state(self, algorithm, epoch: int, metric_value: float, is_best: bool, norm_stats: dict = None):
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
                'numpy': _serialize_numpy_rng_state(np.random.get_state()),
                'python': random.getstate()
            }
        }
        
        # Always save latest for resume
        torch.save(checkpoint, self.outdir / 'latest_checkpoint.pt')
        
        if not is_best:
            return

        # File path for this candidate
        ckpt_path = self.outdir / f"checkpoint_epoch_{epoch:03d}_metric_{metric_value:.4f}.pt"

        entry = {
            "filepath": str(ckpt_path),
            "metric": float(metric_value),
        }


        # -------------------------------
        # CASE 1: fewer than K checkpoints
        # -------------------------------
        if len(self.best_checkpoints) < self.top_k:
            torch.save(checkpoint, ckpt_path)
            self.best_checkpoints.append(entry)

            self.best_checkpoints.sort(
                key=lambda x: x["metric"],
                reverse=(self.mode == "max"),
            )
            self._save_history()
            return

        # ------------------------------------
        # CASE 2: compare against current worst
        # ------------------------------------
        worst = self.best_checkpoints[-1]

        if not self._is_better(metric_value, worst["metric"]):
            # New checkpoint is not good enough → discard
            return

        # Replace worst checkpoint
        worst_path = Path(worst["filepath"])
        if worst_path.exists():
            worst_path.unlink()

        torch.save(checkpoint, ckpt_path)
        self.best_checkpoints[-1] = entry

        # Restore sorted order
        self.best_checkpoints.sort(
            key=lambda x: x["metric"],
            reverse=(self.mode == "max"),
        )

        self._save_history()
        assert 0 < len(self.best_checkpoints) <= self.top_k


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
    
def get_checkpoint(trained_model_dir, device):
    checkpoints_dir = trained_model_dir / 'checkpoints'

    history_file = checkpoints_dir / "checkpoint_history.json"
    
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
            
        if history:
            # Your Checkpointer sorts the list so the best model is always at index 0
            best_checkpoint = history[0]
            weights_path = Path(best_checkpoint['filepath'])
            logger.info(f"Auto-detected best checkpoint from history: {weights_path.name} (Metric: {best_checkpoint['metric']:.4f})")
        else:
            raise ValueError("checkpoint_history.json is empty!")
    else:
        logger.info("No checkpoint_history.json found. Falling back to latest_checkpoint.pt")
        weights_path = checkpoints_dir / 'latest_checkpoint.pt'

    assert weights_path.exists(), f"Weights file not found at {weights_path}"
    try:
        checkpoint = torch.load(weights_path, map_location=device)
    except Exception as e:
        logger.warning(f"torch.load failed with default settings: {e}. Retrying with weights_only=False (legacy checkpoint support).")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    return checkpoint