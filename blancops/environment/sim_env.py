
"""Forward-simulation environment driven by explicit date strings.
 
Renamed from `TestBlancoEnv` to make the role unambiguous (the previous
name was easy to confuse with pytest fixtures or test-time evaluation).
"""
from __future__ import annotations
 
from datetime import datetime, timezone
from typing import Optional
 
import numpy as np
 
from blancops.environment.base import StateSnapshot
from blancops.environment.offline_base import OfflineBlancoEnv
from blancops.data.features.glob_features import calc_twilight
 
import logging
logger = logging.getLogger(__name__)
 
class SimBlancoEnv(OfflineBlancoEnv):
    """Multi-night forward simulation from explicit date strings.
 
    Accepts an optional initial visit history (typically from a previous
    operational night). On subsequent nights the visit counters carry
    forward — each night only resets the clock, not the visit state.
    """
 
    def __init__(
        self,
        *,
        cfg,
        constraints_cfg,
        lookups,
        z_score_stats,
        rel_norm_stats,
        observing_night_strs: list[str],
        initial_counts: Optional[np.ndarray] = None,
    ):
        # Parse before super so max_nights is known in time
        self._night_info = self._parse_night_strs(observing_night_strs)
        super().__init__(
            cfg=cfg,
            constraints_cfg=constraints_cfg,
            lookups=lookups,
            z_score_stats=z_score_stats,
            rel_norm_stats=rel_norm_stats,
            max_nights=len(self._night_info),
        )
        self._initial_counts = initial_counts
        
        self._validate_feature_config()

 
    @staticmethod
    def _parse_night_strs(night_strs: list[str]) -> list[tuple[datetime, str]]:
        """Parse strings like '2026-06-23-half1' or '2026-06-23-full'."""
        parsed = []
        for s in night_strs:
            parts = s.split("-")
            dt = datetime.strptime(
                "-".join(parts[:3]), "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)
            parsed.append((dt, parts[-1]))
        return parsed
 
    # -----------------------------------------------------------------------
    # OfflineBlancoEnv hooks
    # -----------------------------------------------------------------------
 
    def _get_night_config(self, night_idx: int) -> dict:
        night_dt, portion = self._night_info[night_idx]
        ts = night_dt.timestamp()
        sunset = calc_twilight(ts, "set", self.sun_el_limit)
        sunrise = calc_twilight(ts, "rise", self.sun_el_limit)
 
        start_ts, end_ts = sunset, sunrise
        if portion == "half1":
            end_ts = sunset + (sunrise - sunset) / 2
        elif portion == "half2":
            start_ts = sunset + (sunrise - sunset) / 2
 
        return {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "sunset_ts": sunset,
            "sunrise_ts": sunrise,
        }
 
    def _build_night_start_snapshot(self, night_idx: int) -> StateSnapshot:
        cfg = self._get_night_config(night_idx)
 
        if night_idx == 0:
            # Seed counters from the constructor kwarg if provided; else
            # leave counts_cur=None so the tracker keeps the zero state
            # set by reset()'s zero_counts() call.
            counts_cur = (
                self._initial_counts.copy()
                if self._initial_counts is not None
                else None
            )
            return StateSnapshot(
                timestamp=cfg["start_ts"],
                counts_cur=counts_cur,
            )
 
        # Subsequent nights: carry forward; counts_cur=None tells
        # _apply_state_snapshot to skip the tracker write.
        return StateSnapshot(timestamp=cfg["start_ts"])
