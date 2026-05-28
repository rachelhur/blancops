
"""Live operational environment driven by hardware telemetry.
 
Branches off `BaseBlancoEnv` directly rather than going through
`OfflineBlancoEnvBase` because the multi-night scaffolding doesn't
apply: an online episode is a single open-ended night driven by reality,
with time and pointing coming from the telescope rather than simulation.
 
`sync_telemetry` is the public, operator-facing way to align the env
with hardware state. It can be called any time (mid-step, between
actions, on recovery from a weather pause) and is idempotent. Internally
it uses the same `_apply_state_snapshot` primitive that offline envs use
to seed each new night, but with `counts_cur=None` so running visit
history is preserved across syncs — only `_advance_after_action` (via
`_record_visit`) accumulates counts in the live setting.
"""
from __future__ import annotations
 
from typing import Optional
import logging
 
import numpy as np
 
from blancops.environment.base import BaseBlancoEnv, StateSnapshot
from blancops.data.features.glob_features import get_night_boundaries
from blancops.ephemerides import ephemerides
from blancops.configs.constants import (
    WAIT_SIGNAL, ZENITH_FILTER_IDX, FILTER2IDX, IDX2WAVE, FWHM_REF_WAVELENGTH,
)
 
logger = logging.getLogger(__name__)
 
 
class LiveBlancoEnv(BaseBlancoEnv):
    """Real-time environment for live telescope operation.
 
    Constructor is keyword-only. `observing_night` should be a
    timezone-aware datetime; only the date is consulted — twilights are
    computed from it.
    """
 
    def __init__(
        self,
        cfg,
        constraints_cfg,
        lookups,
        z_score_stats,
        rel_norm_stats,
        telemetry_init,
        survey_night_idx=0,
    ):
        self._survey_night_idx = survey_night_idx
        
        super().__init__(
            cfg=cfg,
            constraints_cfg=constraints_cfg,
            lookups=lookups,
            z_score_stats=z_score_stats,
            rel_norm_stats=rel_norm_stats,
        )
        # airmass_limit and sun_el_limit are stored on self by base.
        # Live sessions always start fresh; offline envs load this from
        # historical snapshots via offline_base.py._apply_state_snapshot.
        self._ot_at_sunset = 0.0
        self.sync_telemetry(telemetry=telemetry_init)
        self._validate_feature_config()
    # -----------------------------------------------------------------------
    # Public API: operator-facing telemetry sync
    # -----------------------------------------------------------------------
 
    def sync_telemetry(self, telemetry: Optional[dict] = None) -> None:
        """Align internal state with current hardware state.
 
        Idempotent. Safe to call mid-episode whenever the simulator and     
        the telescope have drifted (manual override, weather pause,
        operator intervention). Visit counters are NOT reset — the
        running per-night history is preserved by omitting `counts_cur`
        from the snapshot.
 
        Args
        ----
        telemetry : Optional[dict]
            Expected keys: 'time' (unix ts), 'ra' / 'dec' (matching the
            unit convention of `self._ra_arr` / `self._dec_arr`), and
            optionally 'filter_idx'. If None, calls
            `self._telemetry.read()`.
        """
        if telemetry is not None:
            filter_idx = telemetry.get(
                "filter_idx", FILTER2IDX.get(telemetry.get("filter"), ZENITH_FILTER_IDX)
            )
            snap = StateSnapshot(
                timestamp=telemetry["timestamp"],
                field_id=self._match_pointing_to_fid(ra=telemetry["ra"], dec=telemetry["dec"]),
                filter_idx=filter_idx,
                # counts_cur intentionally omitted — preserve running history.
            )
            self._apply_state_snapshot(snap)

            # Re-anchor the closed-loop seeing estimate on the real reading.
            # Each chunk rollout then projects forward from this last-telemetry
            # FWHM (see `_get_fwhm`). Syncs without 'fwhm' preserve the prior
            # anchor, mirroring the counts_cur=None visit-history behaviour.
            if telemetry.get("fwhm") is not None:
                self._set_fwhm_reference(
                    fwhm=telemetry["fwhm"],
                    ra=telemetry["ra"],
                    dec=telemetry["dec"],
                    timestamp=telemetry["timestamp"],
                    filter_idx=filter_idx,
                )

        self._refresh_night_boundaries()

        # Fail fast with an actionable message rather than letting the missing
        # feature surface as a downstream NaN in _calculate_global_features.
        if "fwhm" in self.global_feature_names and self._fwhm_ref is None:
            raise ValueError(
                "LiveBlancoEnv: 'fwhm' is a configured global feature but no "
                "telemetry reading has supplied an 'fwhm' value to anchor the "
                "closed-loop estimate. Include 'fwhm' in telemetry_init."
            )

        # Recompute observation arrays so downstream agents see fresh
        # state even if no `step` is called between the sync and the
        # next decision.
        self._update_action_masks()
        self._global_state = self._calculate_global_features()
        if self.include_bin_features:
            self._bin_state = self._calculate_bin_features()

    def record_visit(self, obs_row) -> None:
        """Record that an observation was submitted to the telescope.

        Called by the orchestrator immediately after each hardware submission so
        that subsequent rollouts reflect the in-progress observation.  Unlike
        ``compute_post_action_snapshot``, the visit OT is anchored to the
        current real timestamp (``self._ts``), not a simulated future time —
        this avoids the clock-skew that produced negative staleness values.

        Does not advance ``_ts``; timestamp is only updated via
        ``sync_telemetry`` from real hardware telemetry.
        """
        field_id = int(obs_row["field_id"])
        filter_idx = int(FILTER2IDX[obs_row["filter"]])
        self._record_visit(field_id=field_id, filter_idx=filter_idx)
        self._field_id = field_id
        self._filter_idx = filter_idx
        self._update_action_masks()
        self._global_state = self._calculate_global_features()
        if self.include_bin_features:
            self._bin_state = self._calculate_bin_features()

    def save_snapshot(self) -> StateSnapshot:
        """Capture the full mutable state — pointing, time, and visit counts."""
        return StateSnapshot(
            timestamp=self._ts,
            field_id=self._field_id,
            bin_num=self._bin_num,
            filter_idx=self._filter_idx,
            counts_cur=self._survey_progress_tracker.raw_counts.copy(),
            last_visit_ot_cur=self._last_visit_ot.copy(),
        )

    def restore_snapshot(self, snap: StateSnapshot) -> None:
        """Restore mutable state from a snapshot and recompute derived arrays."""
        self._apply_state_snapshot(snap)
        self._refresh_night_boundaries()
        self._update_action_masks()
        self._global_state = self._calculate_global_features()
        if self.include_bin_features:
            self._bin_state = self._calculate_bin_features()

    # -----------------------------------------------------------------------
    # BaseBlancoEnv lifecycle hooks
    # -----------------------------------------------------------------------
 
    def reset(self, **kwargs):
        raise RuntimeError(
            "LiveBlancoEnv.reset() must not be called mid-session. "
            "Use sync_telemetry() to realign with hardware state, or "
            "construct a new LiveBlancoEnv for a fresh session."
        )

    def _begin_episode(self, ot_at_sunset=0) -> None:
        self._ot_at_sunset = ot_at_sunset
        self.sync_telemetry()
 
    def _advance_after_action(self, action: dict) -> None:
        bin_num = int(action["bin"])
        field_id = int(action["field_id"])
        filter_idx = int(action["filter_idx"])
 
        if bin_num == WAIT_SIGNAL:
            old_ts = self._ts
            self._ts = self._fast_forward()
            logger.info(f"Waited {(self._ts - old_ts) / 60:.1f} minutes")
            # Field/filter unchanged on wait; only bin_num updates below.
            # No visit accumulation — a wait is not an observation.
        else:
            last_field_id = self._field_id
            exptime = self._get_exposure_time(field_id=field_id, filter_idx=filter_idx)
            slew_time = self._get_slew_time(last_field_id, field_id)
            self._ts += exptime + slew_time
 
            # _record_visit lives on BaseBlancoEnv and translates the
            # action's filter_idx to None automatically when the tracker
            # is field-only (1D).
            self._record_visit(field_id=field_id, filter_idx=filter_idx)
 
            self._field_id = field_id
            self._filter_idx = filter_idx
 
        self._bin_num = bin_num
        # Online runs a single open-ended night; never set _is_new_night
        # to True after the initial reset.
        self._is_new_night = False
 
    def _episode_terminated(self) -> bool:
        # Open-ended; ends only at sunrise. Operator-driven termination
        # (weather, hardware fault) is handled outside the env.
        return self._ts >= self._sunrise_ts
 
    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
 
    def _refresh_night_boundaries(self) -> None:
        """Recompute sunrise/sunset for the configured observing night."""
        self._sunset_ts, self._sunrise_ts = get_night_boundaries(
            self._ts, sun_el_limit=self.sun_el_limit
        )
        self._night_end_ts = self._sunrise_ts
 
    def _match_pointing_to_fid(self, ra: float, dec: float) -> int:
        """Return the field ID closest to the given (ra, dec).
 
        Uses the spherical law of cosines on cached field arrays — fine
        for non-pole targets and avoids importing astropy here. Units
        must match `self._ra_arr` / `self._dec_arr` (radians, by
        convention elsewhere in the env).
        """
        cos_sep = (
            np.sin(self._dec_arr) * np.sin(dec)
            + np.cos(self._dec_arr) * np.cos(dec)
            * np.cos(self._ra_arr - ra)
        )
        return int(self._fids[int(np.argmax(cos_sep))])
 
    def _fast_forward(self, step_seconds: float = 60.0) -> float:
        """Advance the clock until at least one valid action exists.
 
        Steps in `step_seconds` increments and re-runs the action mask
        until any field becomes observable, capped at sunrise so we
        never wait past the night. `self._ts` is mutated as a side
        effect during iteration so that `_update_action_masks` (which
        reads `self._ts`) sees the trial time.
        """
        while self._ts < self._sunrise_ts:
            self._ts += step_seconds
            mask = self._update_action_masks()
            if np.any(mask):
                return self._ts
        return self._sunrise_ts


    def _set_fwhm_reference(self, fwhm, ra, dec, timestamp, filter_idx) -> None:
        """Anchor the closed-loop seeing estimate on a real telemetry reading.

        Stores the measured FWHM together with the airmass and filter
        wavelength it was observed at, so in-chunk pointings can be rescaled by
        `_project_fwhm`. Airmass uses the same plane-parallel convention as
        `compute_global_pointing_features` (1/sin(el), elevation clipped).
        """
        _, el = ephemerides.equatorial_to_topographic(ra=ra, dec=dec, time=timestamp)
        el = max(min(el, np.pi / 2), 0.0)
        self._fwhm_ref = float(fwhm)
        self._fwhm_ref_airmass = 1.0 / np.cos(np.pi / 2 - el)
        self._fwhm_ref_wave = IDX2WAVE.get(int(filter_idx), FWHM_REF_WAVELENGTH)

    # -----------------------------------------------------------------------
    # Feature-context hook overrides
    # -----------------------------------------------------------------------

    def _get_fwhm(
        self, timestamp: float, airmass: Optional[float] = None,
        filter_idx: Optional[int] = None,
    ) -> Optional[float]:
        # Project the last-telemetry seeing onto the pointing being evaluated.
        return self._project_fwhm(airmass, filter_idx)

    def _get_t_survey(self) -> Optional[float]:
        s_night_idx = self._get_survey_night_idx()
        s_night_tot = self._get_survey_nights_total()
        return s_night_idx / s_night_tot
 
    def _get_survey_nights_total(self) -> Optional[int]:
        return 2
 
    def _get_survey_night_idx(self) -> Optional[int]:
        return self._survey_night_idx

    @property
    def _night_idx(self):
        return self._survey_night_idx