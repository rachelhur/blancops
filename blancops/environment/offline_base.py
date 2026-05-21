"""Offline (multi-night, simulated-time) environment base.
 
Captures everything `HistoricBlancoEnv` and `SimBlancoEnv` share: the night
counter, simulated time advance (exptime + slew), the night-boundary
lifecycle, and the offline termination condition. Concrete subclasses only
need to say where night boundaries come from and how to seed each night's
initial state.
"""

from __future__ import annotations
 
from abc import abstractmethod
 
import numpy as np
 
from blancops.environment.base import BaseBlancoEnv, StateSnapshot
from blancops.configs.constants import WAIT_SIGNAL
 
import logging
logger = logging.getLogger(__name__)
 
class OfflineBlancoEnvBase(BaseBlancoEnv):
    """Abstract base for envs with a known multi-night schedule.
 
    Subclasses implement two hooks:
      * `_get_night_config(idx)` — returns timing dict for a night
      * `_build_night_start_snapshot(idx)` — returns a `StateSnapshot`
        to seed that night's initial state
    """
 
    def __init__(self, *, max_nights: int, **kwargs):
        super().__init__(**kwargs)
        self.max_nights = max_nights
 
    # -----------------------------------------------------------------------
    # Abstract hooks specific to multi-night scheduling
    # -----------------------------------------------------------------------
 
    @abstractmethod
    def _get_night_config(self, night_idx: int) -> dict:
        """Return {'start_ts', 'end_ts', 'sunset_ts', 'sunrise_ts'} (and
        optionally 'field_id', 'filter_idx', 'bin_num' for the night's
        seed pointing)."""

 
    @abstractmethod
    def _build_night_start_snapshot(self, night_idx: int) -> StateSnapshot:
        """Construct the snapshot to seed `night_idx`'s initial state.
 
        Historic loads recorded visits per night; Sim either zeros (night 0
        with no seed) or returns a snapshot that only sets the timestamp,
        leaving visit counters to carry forward.
        """
 
    # -----------------------------------------------------------------------
    # Concrete implementations of BaseBlancoEnv lifecycle hooks
    # -----------------------------------------------------------------------
 
    def _begin_episode(self) -> None:
        self._night_idx = -1
        self._start_new_night()
        print(f"OfflineBlancoEnvBase._begin_episode(): Night 0 self._ot_at_sunset = {self._ot_at_sunset}")
 
    def _advance_after_action(self, action: dict) -> None:
        bin_num = int(action["bin"])
        field_id = int(action["field_id"])
        filter_idx = int(action["filter_idx"])
 
        # Offline doesn't honor WAIT_SIGNAL — historical data and date-string
        # simulation both have a fixed schedule the agent must drive through.
        # If a wait action arrives here, treat it as a no-op step rather
        # than crashing; log so it's visible in evaluation traces.
        if bin_num == WAIT_SIGNAL:
            logger.debug("Offline env received WAIT_SIGNAL; advancing minimal exptime.")
            # No filter_idx — defaults to the 90 s WAIT fallback.
            self._ts += self._get_exposure_time(field_id=self._field_id)
        else:
            last_field_id = self._field_id
            exptime = float(self._get_exposure_time(field_id=field_id, filter_idx=filter_idx))
            slew_time = float(self._get_slew_time(last_field_id, field_id))
            self._ts += exptime + slew_time
            
            # _record_visit() lives on BaseBlancoEnv, and translates the action's filter_idx
            # to None automatically when tracker is field-only (1D)
            self._record_visit(field_id=field_id, filter_idx=filter_idx)
 
            self._field_id = field_id
            self._filter_idx = filter_idx
 
        self._bin_num = bin_num
 
        # Roll into next night if we've crossed sunrise / scheduled end
        if self._ts >= min(self._sunrise_ts, self._night_end_ts):
            if self._night_idx + 1 < self.max_nights:
                self._start_new_night()
                self._is_new_night = True
                return
        self._is_new_night = False
 
    def _episode_terminated(self) -> bool:
        last_night_done = (
            self._night_idx >= self.max_nights - 1
            and self._ts >= self._sunrise_ts
        )
        all_visited = self._survey_progress_tracker.check_completion()
        return last_night_done or all_visited
 
    # -----------------------------------------------------------------------
    # Internal: night-boundary lifecycle
    # -----------------------------------------------------------------------
 
    def _start_new_night(self) -> None:
        """Advance to the next night and seed its initial state."""
        self._night_idx += 1
        cfg = self._get_night_config(self._night_idx)
        self._sunset_ts = cfg["sunset_ts"]
        self._sunrise_ts = cfg["sunrise_ts"]
        self._night_end_ts = cfg["end_ts"]
        if "ot_at_sunset" not in cfg:
            raise KeyError(
                f"{type(self).__name__}._get_night_config(...) must return "
                f"'ot_at_sunset'; got keys={list(cfg)}. This is required by "
                f"_record_visit's OT-clock bookkeeping."
            )
        self._ot_at_sunset = cfg["ot_at_sunset"]
 
        self._apply_state_snapshot(
            self._build_night_start_snapshot(self._night_idx)
        )
 
        logger.info(
            f"Night {self._night_idx+1}/{self.max_nights}: "
            f"start_ts={self._ts}, sunset={self._sunset_ts}, "
            f"sunrise={self._sunrise_ts}, end={self._night_end_ts}, "
            f"ot_at_sunset={self._ot_at_sunset}"
        )
