from pydantic import BaseModel, Field, computed_field, model_validator
import yaml
from pathlib import Path
from typing import List, Union, Literal
import numpy as np

from typing import Optional
from blancops.configs.enums import *
from blancops.configs.constants import TRAIN_DATA_PATH, BIN_FEATURES
from blancops.data.constants import FILTER2IDX

class DatasetConfig(BaseModel):
    name: str = 'des-data-v0'
    path: str = TRAIN_DATA_PATH
    # cache_in_memory: bool = False
    
    # Data configuration
    nside: int = 16
    action_space: str
    years: List[int] = [2013, 2014, 2015, 2016, 2017, 2018, 2019] # full set of data
    months: List[int] = [i+1 for i in range(12)]
    days: List[int] = [i+1 for i in range(31)]
    filters: List[str] = [filt for filt in FILTER2IDX.keys()]
    
    # Normalization configuration 
    do_cyclical_norm: bool = True
    do_sin_norm: bool = True
    do_log_norm: bool = True
    do_fractional_norm: bool = True
    do_local_mean_z_score: bool = True
    do_z_score_norm: bool = True

    # Configurations calculated after data processing (required for model instantiation)
    state_dim: Optional[int] = None
    bin_state_dim: Optional[int] = None
    num_bins: Optional[int] = None
    num_filters: Optional[int] = None
    num_actions: Optional[int] = None

    # Configurations required for validation
    train_nights: Optional[List[str]] = None
    val_nights: Optional[List[str]] = None
    
    train_val_split: float  = 0.9
    
    # Features
    global_features: List[str]
    bin_features: List[str]
    
    @model_validator(mode='after')
    def validate_features(self) -> 'DatasetConfig':
        for bin_feat in self.bin_features:
            if bin_feat not in BIN_FEATURES:
                raise ValueError(f"{bin_feat} is not implemented.")
        return self
    
class NormalizationConfig(BaseModel): # Not yet done
    feature_names:  Optional[List[str]] = None
    
class BaseModelConfig(BaseModel):
    network: Network = Network.CONTEXTUAL_SCORE_MLP
    loss_strategy: LossStrategy = LossStrategy.PURE_JOINT
    hidden_dim: int = 128
    nlayers: int = 4
    loss_function: str
    contextual_gating: bool = False
    activation: str = "relu"
    
class BCModelConfig(BaseModelConfig):
    algorithm: Literal[Algorithm.BC]

class DDQNModelConfig(BaseModelConfig):
    algorithm: Literal[Algorithm.DDQN]
    tau: float = 0.005 # DDQN specific parameter
    gamma: float = 0.99 # DDQN specific parameter
    
class RewardConfig(BaseModel):
    reward_choice: str = 'expert_actions'
    
class TrainConfig(BaseModel):
    max_epochs: int = 50
    batch_size: int   = 1024
    lr_scheduler: str = "cosine_annealing"
    num_workers: int = 0
    lr_init: float = .001
    lr_final: float = 1e-5
    lr_sched_epoch_start: int = 10
    lr_sched_epoch_duration: int = 30
    patience: int = 20
    device:         str   = "cuda"
    seed:           int   = 42
    
    lr_scheduler_kwargs: Optional[dict] = None
    
    @model_validator(mode='after')
    def validate_lr_scheduler(self):
        # This will now run automatically when model_copy adds the kwargs
        if self.lr_scheduler_kwargs:
            assert self.max_epochs - self.lr_sched_epoch_start - self.lr_sched_epoch_duration >= 0, "The number of epochs must be greater than lr_scheduler_epoch_start + lr_scheduler_dur_epochs"
        return self
    
AnyModelConfig = Union[BCModelConfig, DDQNModelConfig]

class ExperimentConfig(BaseModel):
    experiment_name: str
    experiments_directory: str
    experiment_outdir: Optional[str] = None
    data: DatasetConfig
    model: AnyModelConfig = Field(discriminator="algorithm")
    reward: RewardConfig
    train: TrainConfig
    device: str = 'cuda'
    
    @computed_field
    @property
    def experimentoutdir(self) -> str:
        if self.experiment_outdir is not None:
            return self.experiment_outdir
        else:
            return str(Path(self.experiments_directory) / self.experiment_name)

def load_and_validate(yaml_path: str | Path) -> ExperimentConfig:
    with open(yaml_path, "r") as f:
        raw_data = yaml.safe_load(f)
    return ExperimentConfig(**raw_data) # Added dictionary unpacking

def resolve_and_save(cfg: ExperimentConfig, dataset_dims: dict, dataset_feature_names: dict, lr_scheduler_kwargs: dict, outdir: str | Path) -> ExperimentConfig:
    # Use model_copy to update nested models cleanly
    data_updates = {
        "state_dim": int(dataset_dims['state_dim']),
        "bin_state_dim": int(dataset_dims['bin_state_dim']),
        "num_bins": int(dataset_dims['num_bins']),
        "num_filters": int(dataset_dims['num_filters']),
        "num_actions": int(dataset_dims['num_actions']),
        "global_features": dataset_feature_names['global_features'],
        "bin_features": dataset_feature_names['bin_features'],
    }
    updated_data = cfg.data.model_copy(update=data_updates)
    
    train_updates = {"lr_scheduler_kwargs": lr_scheduler_kwargs}
    updated_train = cfg.train.model_copy(update=train_updates)
    
    resolved_cfg = cfg.model_copy(update={
        "data": updated_data,
        "train": updated_train
    })
    
    with open(outdir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(resolved_cfg.model_dump(mode='json'), f, default_flow_style=False, sort_keys=False)
        
    return resolved_cfg