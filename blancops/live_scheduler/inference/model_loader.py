import os
from typing import Tuple
from pydantic import BaseModel
import yaml
import torch
import logging
from pathlib import Path

from blancops.configs.schema import load_and_validate
from blancops.configs.constants import WORKSPACE
from blancops.rl.registry import build_algorithm

import os
import yaml
import torch
import logging
from pathlib import Path

# Import your domain-specific modules
from blancops.configs.schema import load_and_validate
from blancops.rl.registry import build_algorithm
from blancops.data.lookup import LookupTables
from blancops.rl.agent import Agent

logger = logging.getLogger(__name__)

class DeploymentAgentLoader:
    def __init__(self, base_model_dir: str = None):
        
        self.base_dir = Path(
            base_model_dir or WORKSPACE / "deployable_models"
            )
        self.alias_file = self.base_dir / "aliases.yaml"
        self.aliases = self._load_aliases()

    def _load_aliases(self) -> dict:
        if self.alias_file.exists():
            with open(self.alias_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _resolve_model_dir(self, model_name_or_alias: str) -> Path:
        target_directory_name = self.aliases.get(model_name_or_alias, model_name_or_alias)
        model_path = self.base_dir / target_directory_name
        
        if not model_path.is_dir():
            raise FileNotFoundError(f"Resolved model directory not found: {model_path}")
        return model_path

    def build_agent(
        self, 
        model_path_or_alias: str, 
        device: torch.device, 
        field_lookup_dir: Path, 
        field_choice_method: str
    ) -> Tuple[Agent, BaseModel]:
        model_dir = self._resolve_model_dir(model_path_or_alias)
        
        config_path = model_dir / "config.yaml"
        weights_path = model_dir / "best_weights.pt" # Updated based on your snippet

        logger.info(f"[Model] Building Agent for deployment from: {model_dir}")

        cfg = load_and_validate(config_path)

        lookups = LookupTables().load_from_dir(field_lookup_dir, is_training=False)
        algorithm = build_algorithm(cfg, device).load(weights_path) 
        
        if hasattr(algorithm, 'eval'):
            algorithm.eval()
        elif hasattr(algorithm, 'policy'):
            algorithm.policy.eval()

        agent = Agent(
            algorithm=algorithm, 
            cfg=cfg, 
            lookups=lookups, 
            field_choice_method=field_choice_method
        )
        
        logger.info("[Model] Agent successfully built and ready for scheduling.")
        return agent, cfg