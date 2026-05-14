"""Validation environment driven by recorded survey nights."""
from __future__ import annotations

from typing import Optional
 
import numpy as np
 
from blancops.environment.base import StateSnapshot
from blancops.data.features.glob_features import calc_twilight, get_night_boundaries
 
import logging

from blancops.environment.offline_base import OfflineBlancoEnvBase
logger = logging.getLogger(__name__)

class HistoricBlancoEnv(OfflineBlancoEnvBase):
    """Validation against historically observed nights.
 
    Driven by a pandas groupby keyed on night. Each night's initial visit
    state comes from `lookups.night2fid_visit_hist[night_id]`; per-night
    seeing splines and survey-time arrays are passed at construction so
    the corresponding feature hooks can return real values.
    """
 
    def __init__(
        self,
        *,
        cfg,
        constraints_cfg,
        lookups,
        z_score_stats,
        rel_norm_stats,
        global_pd_nightgroup,
        zenith_bin_states: Optional[np.ndarray] = None,
        t_survey_arr: Optional[np.ndarray] = None,
        fwhm_night_interps: Optional[list] = None,
        survey_nights_total: Optional[int] = None,
    ):
        super().__init__(
            cfg=cfg,
            constraints_cfg=constraints_cfg,
            lookups=lookups,
            z_score_stats=z_score_stats,
            rel_norm_stats=rel_norm_stats,
            max_nights=global_pd_nightgroup.ngroups,
        )
        self._groupbynight = global_pd_nightgroup
        self._night_keys = list(global_pd_nightgroup.groups.keys())
        self._zenith_bin_states = zenith_bin_states
        
        # Per-night context arrays, indexed by self._night_idx.
        self._t_survey_arr = t_survey_arr
        self._survey_night_idx = 0 # is set per-night in _get_night_config
        self._fwhm_night_interps = fwhm_night_interps
        self._survey_nights_total = survey_nights_total
        self._survey_night_idx = 0  # set per-night in _get_night_config
 
        if "t_survey" in self.global_feature_names and self._t_survey_arr is None:
            raise ValueError("HistoricBlancoEnv: 't_survey' configured but t_survey_arr=None")
        if "fwhm" in self.global_feature_names and self._fwhm_night_interps is None:
            raise ValueError("HistoricBlancoEnv: 'fwhm' configured but fwhm_night_interps=None")

        self._validate_feature_config()

    # -----------------------------------------------------------------------
    # OfflineBlancoEnv hooks
    # -----------------------------------------------------------------------
 
    def _get_night_config(self, night_idx: int) -> dict:
        night_df = self._groupbynight.get_group(self._night_keys[night_idx])
        first_row = night_df.iloc[0]
        last_row = night_df.iloc[-1]

        sunset_ts, sunrise_ts = get_night_boundaries(first_row["night"], self.sun_el_limit)
        # sunset = calc_twilight(first_row["night"], "set", horizon=str(self.sun_el_limit))
        # sunrise = calc_twilight(first_row["night"], "rise", horizon=str(self.sun_el_limit))

        # Capture the survey-wide night index recorded in the data; used
        # by the urgency feature.
        self._survey_night_idx = first_row.get("night_idx", 0)
        
        field_id = int(first_row["field_id"])
        filter_idx = int(first_row["filter_idx"])
        bin_num = int(first_row["bin"])

        return {
            "start_ts": first_row["timestamp"],
            "end_ts": last_row["timestamp"],
            "sunset_ts": sunset_ts,
            "sunrise_ts": sunrise_ts,
            "field_id": field_id,
            "filter_idx": filter_idx,
            "bin_num": bin_num
        }
 
    def _build_night_start_snapshot(self, night_idx: int) -> StateSnapshot:
        night_id = self._night_keys[night_idx]
        night_cfg = self._get_night_config(night_idx)

        # Pre-computed zenith bin state, if provided. We set this directly
        # rather than going through the snapshot because bin state isn't
        # part of the StateSnapshot contract — bin features are recomputed
        # by `step()` / `reset()` after `_apply_state_snapshot`.
        if self._zenith_bin_states is not None and self.include_bin_features:
            self._bin_state = self._zenith_bin_states[night_idx]
        history_lookup = self.lookups.night2fidfilt_visit_hist \
                            if self.do_filt \
                            else self.lookups.night2fid_visit_hist
                            
        print
        
        return StateSnapshot(
            timestamp=night_cfg["start_ts"],
            field_id=night_cfg["field_id"],
            bin_num=night_cfg["bin_num"],
            filter_idx=night_cfg["filter_idx"],
            counts_cur = history_lookup[night_id].copy()
        )

    # -----------------------------------------------------------------------
    # Feature-context hook overrides
    # -----------------------------------------------------------------------
 
    def _get_t_survey(self) -> Optional[float]:
        if self._t_survey_arr is None:
            return None
        return float(self._t_survey_arr[self._night_idx])
 
    def _get_fwhm(self, timestamp: float) -> Optional[float]:
        if self._fwhm_night_interps is None:
            return None
        return float(self._fwhm_night_interps[self._night_idx](timestamp))
 
    def _get_survey_nights_total(self) -> Optional[int]:
        return len(self.lookups.night2fid_visit_hist)
 
    def _get_survey_night_idx(self) -> Optional[int]:
        return self._survey_night_idx

            