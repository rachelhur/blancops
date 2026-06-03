"""
Permutation importance for trained BC policies via KL divergence.

Typical per-feature usage::

    df = permutation_importance_from_run("/path/to/run_dir")

For group-level or abs/rel-pair analysis, extract tensors first then call the
lower-level functions directly::

    cache = ValDatasetCache.load(run_dir / "checkpoints" / "val_dataset_cache.pt")
    global_obs, bin_obs, action_masks = obs_tensors_from_cache(cache)
    df = compute_group_permutation_importance(policy, global_obs, bin_obs,
                                              action_masks, FEATURE_GROUPS, ...)

For each feature (or group):
    1. Compute nominal log-probs P(a | s) across all val observations.
    2. Shuffle the feature's values across observations, breaking its signal
       while leaving all other features intact.
    3. Recompute Q(a | s_permuted) and measure mean KL(P || Q).

Large KL  ->  policy distribution shifts significantly; feature is load-bearing.
Near-zero ->  feature is ignored or fully compensated by a correlated partner.

CORRELATED FEATURE CAVEAT
Individual permutation understates importance for correlated pairs such as
(ha, airmass), (ha_global, ha_bin), and (abs_feat, rel_feat). When feature A
is permuted, the model partially recovers via correlated feature B, so A's KL
is depressed. Use compute_group_permutation_importance for groups like
'ephemeris' or 'survey_depth', permuting them jointly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from blancops.configs.rl_schema import load_and_validate
from blancops.data.feature_cache import ValDatasetCache
from blancops.rl.agent_factory import AgentFactory

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def obs_tensors_from_cache(
    cache: ValDatasetCache,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract per-transition (global_obs, bin_obs, action_masks) from a cache.

    Returns CPU tensors of shape:
        global_obs   (N_transitions, D_glob)
        bin_obs      (N_transitions, n_bins, D_bin)
        action_masks (N_transitions, n_actions)
    """
    c_idxs = torch.from_numpy(cache.curr_compact_idxs).long()
    return (
        cache.states[c_idxs].cpu(),
        cache.bin_states[c_idxs].cpu(),
        cache.action_masks[c_idxs].cpu(),
    )


@torch.no_grad()
def _batch_log_probs(
    policy,
    global_obs: torch.Tensor,    # (N, D_glob) CPU
    bin_obs: torch.Tensor,       # (N, n_bins, D_bin) CPU
    action_masks: torch.Tensor,  # (N, n_actions) CPU
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:               # (N, A) CPU
    chunks = []
    for i in range(0, len(global_obs), batch_size):
        g    = global_obs[i:i+batch_size].to(device, dtype=torch.float32)
        b    = bin_obs[i:i+batch_size].to(device, dtype=torch.float32)
        mask = action_masks[i:i+batch_size].to(device)

        logits    = policy.core_net(g, b)
        mask_val  = torch.finfo(logits.dtype).min
        logits    = logits.masked_fill(~mask.bool(), mask_val)
        log_probs = F.log_softmax(logits, dim=-1)
        chunks.append(log_probs.cpu())
    return torch.cat(chunks)


def _mean_kl(log_p: torch.Tensor, log_q: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean KL(P || Q) over the batch dimension.

    Handles -inf log_q (masked actions) by clamping; those actions have p~0
    so their contribution is negligible.
    """
    p     = log_p.exp().clamp(min=eps)
    log_q = torch.where(torch.isfinite(log_q), log_q, torch.full_like(log_q, np.log(eps)))
    return (p * (log_p - log_q)).sum(dim=-1).mean().item()


def _setup_policy(policy, device: torch.device) -> None:
    if hasattr(policy, "eval"):
        policy.eval()
    if hasattr(policy, "to"):
        policy.to(device)


# ---------------------------------------------------------------------------
# Per-feature permutation importance
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    policy,
    global_obs: torch.Tensor,       # (N, D_glob) CPU
    bin_obs: torch.Tensor,           # (N, n_bins, D_bin) CPU
    action_masks: torch.Tensor,      # (N, n_actions) CPU
    global_feature_names: list[str],
    bin_feature_names: list[str],
    n_permutations: int = 5,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Per-feature permutation importance.

    Each feature is permuted independently n_permutations times. The mean and
    std of KL(P_nominal || P_permuted) are reported. std_kl > mean_kl suggests
    more permutations are needed.

    Returns
    -------
    DataFrame sorted by mean_kl descending.
    Columns: feature, feature_type, mean_kl, std_kl
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _setup_policy(policy, device)

    N = global_obs.shape[0]
    nominal_lp = _batch_log_probs(policy, global_obs, bin_obs, action_masks, batch_size, device)
    rows = []

    for feat_type, names in [("global", global_feature_names), ("bin", bin_feature_names)]:
        logger.info(f"Permuting {len(names)} {feat_type} features ({n_permutations}x each, {N} obs)...")
        for i, name in enumerate(tqdm(names, desc=feat_type, leave=False)):
            kls = []
            for _ in range(n_permutations):
                perm = torch.randperm(N)
                if feat_type == "global":
                    pg = global_obs.clone(); pg[:, i] = global_obs[perm, i]
                    pb = bin_obs
                else:
                    pg = global_obs
                    pb = bin_obs.clone(); pb[:, :, i] = bin_obs[perm, :, i]

                perm_lp = _batch_log_probs(policy, pg, pb, action_masks, batch_size, device)
                kls.append(_mean_kl(nominal_lp, perm_lp))

            rows.append(dict(
                feature=name,
                feature_type=feat_type,
                mean_kl=float(np.mean(kls)),
                std_kl=float(np.std(kls)),
            ))

    return (
        pd.DataFrame(rows)
        .sort_values("mean_kl", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Group permutation importance
# ---------------------------------------------------------------------------

def compute_group_permutation_importance(
    policy,
    global_obs: torch.Tensor,       # (N, D_glob) CPU
    bin_obs: torch.Tensor,           # (N, n_bins, D_bin) CPU
    action_masks: torch.Tensor,      # (N, n_actions) CPU
    feature_groups: dict[str, dict[str, list[int]]],
    global_feature_names: list[str],
    bin_feature_names: list[str],
    n_permutations: int = 5,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Group-level permutation importance with a shared permutation index.

    All features in a group are permuted using the same index, so within-group
    correlations are preserved while the group's signal is broken. This gives
    the correct importance estimate for correlated feature clusters.

    Parameters
    ----------
    feature_groups : dict
        Maps group name -> {
            "global": [column_indices_into_global_obs],
            "bin":    [column_indices_into_bin_obs],
        }

    Returns
    -------
    DataFrame sorted by mean_kl descending.
    Columns: group, global_features, bin_features, mean_kl, std_kl
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _setup_policy(policy, device)

    N = global_obs.shape[0]
    nominal_lp = _batch_log_probs(policy, global_obs, bin_obs, action_masks, batch_size, device)
    rows = []

    for group_name, idx_dict in tqdm(feature_groups.items(), desc="groups", leave=False):
        g_idxs = idx_dict.get("global", [])
        b_idxs = idx_dict.get("bin",    [])
        kls = []

        for _ in range(n_permutations):
            perm = torch.randperm(N)
            pg, pb = global_obs.clone(), bin_obs.clone()
            for gi in g_idxs:
                pg[:, gi]    = global_obs[perm, gi]
            for bi in b_idxs:
                pb[:, :, bi] = bin_obs[perm, :, bi]

            perm_lp = _batch_log_probs(policy, pg, pb, action_masks, batch_size, device)
            kls.append(_mean_kl(nominal_lp, perm_lp))

        rows.append(dict(
            group=group_name,
            global_features=[global_feature_names[i] for i in g_idxs],
            bin_features=[bin_feature_names[i] for i in b_idxs],
            mean_kl=float(np.mean(kls)),
            std_kl=float(np.std(kls)),
        ))

    return (
        pd.DataFrame(rows)
        .sort_values("mean_kl", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Abs vs rel pair comparison
# ---------------------------------------------------------------------------

def compare_abs_rel_pairs(
    policy,
    global_obs: torch.Tensor,       # (N, D_glob) CPU
    bin_obs: torch.Tensor,           # (N, n_bins, D_bin) CPU
    action_masks: torch.Tensor,      # (N, n_actions) CPU
    abs_rel_pairs: list[tuple[str, str, int]],
    bin_feature_names: list[str],
    n_permutations: int = 5,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """For each (abs, rel) bin feature pair, test all four permutation conditions.

    Conditions:
        both_present  -- no permutation; KL = 0 by definition
        neither       -- both abs and rel permuted
        abs_only      -- keep abs, permute rel
        rel_only      -- keep rel, permute abs

    Parameters
    ----------
    abs_rel_pairs : list of (abs_feature_name, rel_feature_name, abs_col_idx)
        abs_col_idx is the column index of the absolute feature in bin_obs.
        The rel feature is looked up by name in bin_feature_names.

    Returns
    -------
    DataFrame with columns: pair, condition, mean_kl
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _setup_policy(policy, device)

    N = global_obs.shape[0]
    nominal_lp = _batch_log_probs(policy, global_obs, bin_obs, action_masks, batch_size, device)
    rows = []

    for abs_name, rel_name, abs_idx in abs_rel_pairs:
        rel_idx = bin_feature_names.index(rel_name)

        for condition, permute_abs, permute_rel in [
            ("both_present", False, False),
            ("neither",      True,  True),
            ("abs_only",     False, True),
            ("rel_only",     True,  False),
        ]:
            if not permute_abs and not permute_rel:
                rows.append(dict(pair=f"{abs_name}/{rel_name}", condition=condition, mean_kl=0.0))
                continue

            kls = []
            for _ in range(n_permutations):
                perm = torch.randperm(N)
                pb = bin_obs.clone()
                if permute_abs:
                    pb[:, :, abs_idx] = bin_obs[perm, :, abs_idx]
                if permute_rel:
                    pb[:, :, rel_idx] = bin_obs[perm, :, rel_idx]
                perm_lp = _batch_log_probs(policy, global_obs, pb, action_masks, batch_size, device)
                kls.append(_mean_kl(nominal_lp, perm_lp))

            rows.append(dict(
                pair=f"{abs_name}/{rel_name}",
                condition=condition,
                mean_kl=float(np.mean(kls)),
            ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def permutation_importance_from_run(
    run_dir: Path,
    n_permutations: int = 5,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Compute per-feature permutation importance for a completed training run.

    Loads from the standard run-directory layout::

        <run_dir>/checkpoints/model.pt
        <run_dir>/checkpoints/val_dataset_cache.pt
        <run_dir>/configs/resolved_config.yaml

    Returns the same DataFrame as compute_permutation_importance.
    """
    run_dir = Path(run_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_and_validate(run_dir / "configs" / "resolved_config.yaml")

    policy, _ = AgentFactory.load_policy(
        weights_path=run_dir / "checkpoints" / "model.pt",
        cfg=cfg,
        device=device,
    )

    cache = ValDatasetCache.load(run_dir / "checkpoints" / "val_dataset_cache.pt")
    global_obs, bin_obs, action_masks = obs_tensors_from_cache(cache)

    return compute_permutation_importance(
        policy=policy,
        global_obs=global_obs,
        bin_obs=bin_obs,
        action_masks=action_masks,
        global_feature_names=cache.global_feature_names,
        bin_feature_names=cache.bin_feature_names,
        n_permutations=n_permutations,
        batch_size=batch_size,
        device=device,
    )

