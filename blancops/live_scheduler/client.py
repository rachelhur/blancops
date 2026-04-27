"""Telescope API adapters for the live scheduler.

This module defines the scheduler-facing telescope interface and provides:

- MockTelescopeClient: local simulation of pointing, slew, and exposure timing.
- BlancoTelescopeClient: placeholder wrapper for the real observatory control path.
"""

from abc import ABC, abstractmethod
from blancops.math import units, geometry
from blancops.ephemerides import ephemerides, time_utils


class TelescopeClient(ABC):
    """Abstract interface for telescope-control interactions used by the scheduler."""

    @abstractmethod
    def get_telemetry(self):
        """
        Return current telemetry needed by scheduling logic.

        Returns
        -------
        dict
            Current telescope state, including current pointing coordinates.
        """

        pass

    @abstractmethod
    def check_exposure_status(self):
        """
        Report whether the current exposure is complete.

        Returns
        -------
        bool
            True when the system is ready for the next submission.
        """

        pass

    @abstractmethod
    def submit_observation(self, obs_row):
        """
        Submit a single observation request to the control system.

        Arguments
        ---------
        obs_row: dict or pandas.Series
            Observation request containing at least RA, Dec, and filter fields.
        """

        pass


class MockTelescopeClient(TelescopeClient):
    """In-memory telescope simulator for development and integration testing."""

    def __init__(self, exposure_duration=90):
        """Initialize mock timing and initial pointing state.

        Arguments
        ---------
        exposure_duration: float [90]
            Simulated exposure time in seconds.
        """

        # model internal state to simulate exposure timing
        self.last_exposure_submit_time = -float("inf")
        self.exposure_duration = exposure_duration
        self.slew_time = 0

        # track current pointing to model stepping through observations
        self.current_ra, self.current_dec = ephemerides.get_source_ra_dec("zenith")

        print("[API] Initialized mock telescope API connection.")

    def get_telemetry(self):
        """Return the currently simulated telescope pointing."""

        return {"pointing_ra": self.current_ra, "pointing_dec": self.current_dec}

    def check_exposure_status(self):
        """Return True when simulated slew+exposure time has elapsed."""

        # compare elapsed wall-clock time against modeled slew + exposure duration
        delta = time_utils.utc_now() - self.last_exposure_submit_time
        return delta > self.slew_time + self.exposure_duration

    def submit_observation(self, obs_row):
        """Submit an observation into the mock queue and update simulator state."""

        # approximate slew time from angular separation between old/new pointings
        angsep = geometry.angular_separation(
            (self.current_ra, self.current_dec), (obs_row["ra"], obs_row["dec"])
        )
        self.slew_time = geometry.blanco_slew_time(angsep) / units.second

        # update internal state to reflect the new observation
        self.last_exposure_submit_time = time_utils.utc_now()
        self.current_ra = obs_row["ra"]
        self.current_dec = obs_row["dec"]

        print(
            f"[API] SUBMITTED: RA={obs_row['ra']}, DEC={obs_row['dec']}, FILTER={obs_row['filter']}"
        )
        print(
            f"[API] Estimated time until ready for next submission: {self.slew_time + self.exposure_duration:.1f}s."
        )


class BlancoTelescopeClient(TelescopeClient):
    """Placeholder for the production telescope control-system integration."""

    def __init__(self):
        """Initialize and confirm the connection to the observatory control system."""

        self.connected = True
        print("[API] Initialized connection to telescope control system.")

    def get_telemetry(self):
        """
        Fetch live telemetry.

        Current placeholder behavior returns zenith coordinates.
        """

        ra, dec = ephemerides.get_source_ra_dec("zenith")
        return {"pointing_ra": ra, "pointing_dec": dec}

    def check_exposure_status(self):
        """
        Return exposure readiness state from control system.

        Current placeholder behavior always returns True. In production, this should
        return True if the system is idle or the exposure is complete, and False if an
        exposure is ongoing.
        """

        # Placeholder: True if idle/done, False if exposing.
        # Mocked to True to simulate immediate completion for testing.
        return True

    def submit_observation(self, obs_row):
        """
        Submit one observation request to the control system.

        Current placeholder behavior prints the queued request.
        """

        # Placeholder: Issue synchronous command to API
        print(
            f"[API] SUBMITTED: RA={obs_row['ra']}, DEC={obs_row['dec']}, FILTER={obs_row['filter']}"
        )
