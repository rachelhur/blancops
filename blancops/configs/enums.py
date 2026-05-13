from enum import Enum

class Algorithm(str, Enum):
    BC = "bc"
    DQN = "dqn"
    DDQN = "ddqn"
    CQL = "cql"

class LossStrategy(str, Enum):
    PURE_JOINT = "pure_joint"
    HYBRID_MARGINAL = "hybrid_marginal"
    PSEUDO_AUTOREGRESSIVE = "pseudo_ar"
    AUTOREGRESSIVE = "autoregressive"
    # MARGINAL = "marginal"
    
class Network(str, Enum):
    MLP = "mlp"
    CONTEXTUAL_SCORE_MLP = "context_score_mlp"
    MULTI_HEAD_MLP = "multi_head_mlp"
    AUTOREGRESSIVE = "autoregressive"

class Reward(str, Enum):
    EXPERT_ACTION = "expert_action"
    SURVEY_UNIFORMITY = "survey_uniformity"

class LookupKeys(str, Enum):
    FIELDS = "fields_table.json"
    TARGET_FIDFILT_COUNTS = "target_counts_per_fidfilt.pkl"
    FIDFILT_EXPTIME = "fidfilt_exptime.pkl"
    TARGET_FILT_COUNTS = "target_counts_per_filter.pkl"
    TARGET_FID_COUNTS = "target_counts_per_fid.pkl"
    
    # TRAIN DATA LOOKUP KEYS    
    TARGET_FID2VISITS_TRAIN = "target_counts_per_fid_train.json"
    TARGET_FID2VISITS_EVAL = "target_counts_per_fid_eval.json"
    NIGHT2FID_VISIT_HIST = "night2fidvisits.pkl"
    NIGHT2FIDFILT_VISIT_HIST = "night2fidfilt_visits.pkl"