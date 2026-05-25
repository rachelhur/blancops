from blancops.configs.rl_schema import ActionConstraints
from blancops.data.features.normalizations import StateNormalizer, build_normalizer, build_normalizer_kwargs
from blancops.environment.live_env import LiveBlancoEnv

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as au

from blancops.math import units

def build_env(cfg, norm_stats, lookups, telemetry_now):
    constraints_cfg = ActionConstraints()
    zscore_stats = norm_stats.get('z_score', {})
    rel_norm_stats = norm_stats.get('rel_norm', {})
    env = LiveBlancoEnv(
        cfg=cfg,
        constraints_cfg=constraints_cfg,
        lookups=lookups,
        z_score_stats=zscore_stats, 
        rel_norm_stats=rel_norm_stats,
        telemetry_init=telemetry_now
    )
    return env

def get_visible_targets(
    target_radecs: np.ndarray = None, 
    obs_time_utc: str = '2026-06-23T04:00:00',
    airmass_limit: float = 1.3
):
    """
    Args
    ----
    target_radecs: (np.ndarray)
        Target radecs in units rad with shape (nfields, 2)
    """
    blanco_loc = EarthLocation.of_site('ctio')
    obs_time = Time(obs_time_utc, scale='utc')

    if target_radecs is None:
        ra_bins = np.linspace(0, 360, 100)
        dec_bins = np.linspace(-80, 20, 50)
        ra_grid, dec_grid = np.meshgrid(ra_bins, dec_bins)
    else:
        ra_grid, dec_grid = target_radecs[:, 0], target_radecs[:, 1]
        ra_grid /= units.deg
        dec_grid /= units.deg
        
    targets = SkyCoord(ra=ra_grid.ravel(), dec=dec_grid.ravel(), unit='deg', frame='icrs')
    altaz_frame = AltAz(obstime=obs_time, location=blanco_loc)
    
    # Transform to azel
    target_altaz = targets.transform_to(altaz_frame)

    # Airmass constraints
    elevation_limit = np.arcsin(1 / airmass_limit) * au.deg 
    # elevation_limit = 30 * au.deg
    is_observable_mask = target_altaz.alt > elevation_limit
    
    # Reshape the mask back to the 2D grid shape if needed for visualization or state representation
    observable_grid = is_observable_mask.reshape(ra_grid.shape)
    
    # You can now filter your RA and Dec arrays using this boolean mask
    valid_ra = ra_grid.ravel()[is_observable_mask]
    valid_dec = dec_grid.ravel()[is_observable_mask]
    return valid_ra, valid_dec