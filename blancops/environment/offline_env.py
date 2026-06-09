"""Forward-simulation environment driven by explicit date strings."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np

from blancops.environment.base import StateSnapshot
from blancops.environment.offline_base import BaseBlancoOfflineEnv
from blancops.data.features.glob_features import calc_twilight, get_night_boundaries
from blancops.configs.constants import ZENITH_AIRMASS, FWHM_REF_WAVELENGTH

import logging
logger = logging.getLogger(__name__)


class OfflineBlancoEnv(BaseBlancoOfflineEnv):
    """Multi-night forward simulation from explicit date strings.

    Accepts optional seeds (counts, last-visit OT timestamps, OT clock at
    sunset of night 0) for continuing a survey mid-stream. With nothing
    seeded, night 0 starts at OT=0 with no prior visits, and the OT
    clock cascades night-to-night at wall-clock rate from sunset to
    sunrise (regardless of half/full portion — the only formula that
    keeps ot_now = ot_at_sunset + (ts - sunset_ts) monotonic across
    half-night transitions).
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
        initial_last_visit_ot: Optional[np.ndarray] = None,
        initial_ot_at_sunset: float = 0.0,
        initial_fwhm: Optional[float] = None,
    ):
        # Parse before super so max_nights is known in time.
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
        self._initial_last_visit_ot = initial_last_visit_ot
        self._initial_ot_at_sunset = float(initial_ot_at_sunset)

        # Seed seeing for the forward simulation. There is no measured
        # seeing in a date-string sim, so `initial_fwhm` is interpreted as the
        # assumed zenith seeing (airmass=1) at the reference wavelength and is
        # projected to each pointing's airmass/filter by `_get_fwhm`. Base
        # atmospheric seeing is held constant across the run.
        if initial_fwhm is not None:
            self._fwhm_ref = float(initial_fwhm)
            self._fwhm_ref_airmass = ZENITH_AIRMASS
            self._fwhm_ref_wave = FWHM_REF_WAVELENGTH

        # Cache prevents double-advancement if _get_night_config
        # is re-entered for a night that's already been started.
        self._night_cfg_cache: dict[int, dict] = {}

        if "fwhm" in self.global_feature_names and self._fwhm_ref is None:
            raise ValueError(
                "OfflineBlancoEnv: 'fwhm' is a configured global feature but "
                "initial_fwhm=None. Pass initial_fwhm (assumed zenith seeing, "
                "arcsec) so the forward sim can project seeing per pointing."
            )

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

    def _begin_episode(self) -> None:
        # Restart OT cascade and night cache on every reset; otherwise
        # state leaks from previous episodes.
        self._night_cfg_cache = {}
        super()._begin_episode()

    def _get_night_config(self, night_idx: int) -> dict:
        if night_idx in self._night_cfg_cache:
            return self._night_cfg_cache[night_idx]

        night_dt, portion = self._night_info[night_idx]
        sunset_ts, sunrise_ts = get_night_boundaries(
            night_dt, self.sun_el_limit - 0.1
        )

        start_ts, end_ts = sunset_ts, sunrise_ts
        if portion == "half1":
            end_ts = sunset_ts + (sunrise_ts - sunset_ts) / 2
        elif portion == "half2":
            start_ts = sunset_ts + (sunrise_ts - sunset_ts) / 2
            
        
        # Anchor ot_at_sunset so that
        #     ot_now @ start_ts  ==  OT clock at the moment we rolled
        #                            over from the previous night.
        # Derivation: ot_now = ot_at_sunset + (ts - sunset_ts), so for the
        # equality to hold at ts=start_ts:
        #     ot_at_sunset = prev_OT_at_rollover - (start_ts - sunset_ts)
        #
        # For half1/full this offset is 0 (start_ts == sunset_ts).
        # For half2, ot_at_sunset gets a NEGATIVE anchor of -half_dur so that
        # ot_now at the mid-night start point picks up exactly where the
        # previous night left off, instead of jumping ahead by half a night.
        if night_idx == 0:
            prev_OT_at_rollover = self._initial_ot_at_sunset
        else:
            prev_OT_at_rollover = (
                self._ot_at_sunset + (self._ts - self._sunset_ts)
            )
        ot_at_sunset = prev_OT_at_rollover - (start_ts - sunset_ts)

        cfg = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "sunset_ts": sunset_ts,
            "sunrise_ts": sunrise_ts,
            "ot_at_sunset": ot_at_sunset,
        }
        self._night_cfg_cache[night_idx] = cfg
        return cfg
    # def _get_night_config(self, night_idx: int) -> dict:
    #     # Idempotent re-read for the same night (diagnostics, tests).
    #     if night_idx in self._night_cfg_cache:
    #         return self._night_cfg_cache[night_idx]

    #     night_dt, portion = self._night_info[night_idx]
    #     # ts = night_dt.timestamp()
    #     sunset_ts, sunrise_ts = get_night_boundaries(night_dt, self.sun_el_limit - .1)

    #     # sunset = calc_twilight(ts, "set", self.sun_el_limit)
    #     # sunrise = calc_twilight(ts, "rise", self.sun_el_limit)

    #     start_ts, end_ts = sunset_ts, sunrise_ts
    #     if portion == "half1":
    #         end_ts = sunset_ts + (sunrise_ts - sunset_ts) / 2
    #     elif portion == "half2":
    #         start_ts = sunset_ts + (sunrise_ts - sunset_ts) / 2

    #     cfg = {
    #         "start_ts": start_ts,
    #         "end_ts": end_ts,
    #         "sunset_ts": sunset_ts,
    #         "sunrise_ts": sunrise_ts,
    #         "ot_at_sunset": self._running_ot_at_sunset,
    #     }
    #     self._night_cfg_cache[night_idx] = cfg
    #     # Advance OT cascade for the next night: full sunset→sunrise
    #     # span, irrespective of half-night portion. See class docstring.
    #     self._running_ot_at_sunset += (end_ts - start_ts)
    #     return cfg

    def _build_night_start_snapshot(self, night_idx: int) -> StateSnapshot:
        cfg = self._get_night_config(night_idx)

        if night_idx == 0:
            # Seed counters and last-visit OT timestamps from constructor
            # kwargs if provided. None means "leave the (already-zeroed-
            # by-reset) state alone" — see base.reset().
            counts_cur = (
                self._initial_counts.copy()
                if self._initial_counts is not None
                else None
            )
            last_visit_ot_cur = (
                self._initial_last_visit_ot.copy()
                if self._initial_last_visit_ot is not None
                else None
            )
            return StateSnapshot(
                timestamp=cfg["start_ts"],
                counts_cur=counts_cur,
                last_visit_ot_cur=last_visit_ot_cur,
            )

        # Subsequent nights: just advance the clock. Tracker and
        # _last_visit_ot carry forward via _apply_state_snapshot's
        # None-skip behaviour.
        return StateSnapshot(timestamp=cfg["start_ts"])

    # -----------------------------------------------------------------------
    # Feature-context hook overrides
    # -----------------------------------------------------------------------

    def _get_fwhm(
        self, timestamp: float, el: Optional[float] = None,
        filter_idx: Optional[int] = None,
    ) -> Optional[float]:
        # Project the seeded zenith seeing onto the current pointing.
        return self._project_fwhm(el, filter_idx)