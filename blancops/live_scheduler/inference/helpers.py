from blancops.configs.rl_schema import ActionConstraints
from blancops.data.features.normalizations import StateNormalizer, build_normalizer, build_normalizer_kwargs
from blancops.environment.live_env import LiveBlancoEnv

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

def build_env(cfg, norm_stats, lookups, telemetry_now):
    
    constraints_cfg = ActionConstraints()
    zscore_stats = norm_stats.get('z_score', {})
    rel_norm_stats = norm_stats.get('rel_norm', {})
    
    global_normalizer = build_normalizer(state_feature_names=cfg.data.global_features, cfg=cfg)
    bin_normalizer = build_normalizer(state_feature_names=cfg.data.bin_features, cfg=cfg)
    norm_kwargs = build_normalizer_kwargs(cfg.data.norm)
    global_normalizer = StateNormalizer(
        state_feature_names=cfg.data.global_features, 
        **norm_kwargs
    )
    bin_normalizer = StateNormalizer(
        state_feature_names=cfg.data.bin_features, 
        **norm_kwargs
    )
    env = LiveBlancoEnv(
        cfg=cfg,
        constraints_cfg=constraints_cfg,
        lookups=lookups,
        global_normalizer=global_normalizer,
        bin_normalizer=bin_normalizer,
        z_score_stats=zscore_stats, 
        rel_norm_stats=rel_norm_stats,
        telemetry_init=telemetry_now
    )
    return env

def get_above_horizon_targets(obs_time, lookups):
    blanco_loc = EarthLocation.of_site('ctio')
    obs_time = '2026-06-24T04:00:00' if obs_time is None else obs_time
    obs_time = Time(obs_time, scale='utc') 

    # Sky grid
    ra_bins = lookups.fields.ra * u.rad
    dec_bins = lookups.fields.dec * u.rad
    ra_grid, dec_grid = np.meshgrid(ra_bins, dec_bins)

    # Flatten
    targets = SkyCoord(ra=ra_grid.ravel(), dec=dec_grid.ravel(), frame='icrs')

    # Azel frame
    altaz_frame = AltAz(obstime=obs_time, location=blanco_loc)

    # Transform targets to azel
    target_altaz = targets.transform_to(altaz_frame)

    # Mask above horizon
    is_observable_mask = target_altaz.alt > 0

    observable_grid = is_observable_mask.reshape(ra_grid.shape)

    valid_ra = ra_grid.ravel()[is_observable_mask]
    valid_dec = dec_grid.ravel()[is_observable_mask]

    print(f"Total targets evaluated: {len(targets)}")
    print(f"Observable targets at {obs_time.iso}: {np.sum(is_observable_mask)}")
