"""Telescope Client adapters for the live scheduler.

This module defines the scheduler-facing telescope interface and provides:

- MockTelescopeClient: local simulation of pointing, slew, and exposure timing.
- BlancoTelescopeClient: placeholder wrapper for the real observatory control path.
"""

from abc import ABC, abstractmethod
from blancops.math import units, geometry
from blancops.ephemerides import ephemerides, time_utils
from blancops.live_scheduler.scl import SCL
import json
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

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

    @abstractmethod
    def check_telemetry_change(self, current_telemetry, last_telemetry):
        """
        Report whether the telescope state has changed meaningfully since the last check.

        Arguments
        ---------
        current_telemetry : dict
            Most recently fetched telemetry dict.
        last_telemetry : dict
            Previously stored telemetry dict.

        Returns
        -------
        bool
            True if the pointing or other state has changed enough to warrant replanning.
        """

        pass

    @abstractmethod
    def close(self):
        """
        Clean up any open connections or resources when the client is no longer needed.
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

        logger.info("[Client] Initialized mock telescope client.")

    def get_telemetry(self):
        """Return the currently simulated telescope pointing."""

        return {"pointing_ra": self.current_ra, "pointing_dec": self.current_dec}

    def check_telemetry_change(self, current_telemetry, last_telemetry):
        """Returns True if the pointing has changed since the last call."""
        if not last_telemetry:
            return True
        tel_current = (current_telemetry.get("pointing_ra"), current_telemetry.get("pointing_dec"))
        tel_last = (last_telemetry.get("pointing_ra"), last_telemetry.get("pointing_dec"))
        return tel_current != tel_last
        
    def check_exposure_status(self):
        """Return True when simulated slew+exposure time has elapsed."""

        # compare elapsed wall-clock time against modeled slew + exposure duration
        delta = time_utils.utc_now() - self.last_exposure_submit_time
        return delta > self.slew_time + self.exposure_duration

    def submit_observation(self, obs_row, exp_time=None):
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
        if exp_time is not None:
            self.exposure_duration = exp_time

        logger.info(
            f"[Client] SUBMITTED: RA={obs_row['ra']}, DEC={obs_row['dec']}, FILTER={obs_row['filter']}"
        )
        logger.info(
            f"[Client] Estimated time until ready for next submission: {self.slew_time + self.exposure_duration:.1f}s."
        )

    def close(self):
        """No resources to clean up for the mock client."""
        logger.info("[Client] Closing mock telescope client (no resources to clean up).")


class BlancoSCLTelescopeClient(TelescopeClient):
    """Blanco telescope control-system integration using SCL network."""

    def __init__(self, propid=None, server_ip="observer4.ctio.noao.edu", server_port=20000):
        """Initialize and confirm the connection to the control system."""

        # Initialize the TCP/IP communication client
        logger.info(f"[Client] Attempting to connect to SCLN server at {server_ip}:{server_port}...")
        self.scl_client = SCL(server_ip, server_port)
        self.transaction_id = 0
        
        # check if connection was successful
        if self.scl_client.is_connected():
            logger.info(f"[Client] Initialized connection to SCLN server at {server_ip}:{server_port}.")
        else:
            logger.warning(f"[Client] WARNING: Could not connect to SCLN server at {server_ip}:{server_port}.")

        # track current pointing based on submissions
        self.current_ra, self.current_dec = None, None
        self.current_time = datetime.now().isoformat(timespec='milliseconds')

        self.propid = propid

    def _build_base_message(self, msg_type):
        """Helper to construct the standard JSON envelope for SCLN messages."""
        if msg_type == "COMMAND":
            self.current_time = datetime.now().isoformat(timespec='milliseconds')
            cmd = {
                "type": "COMMAND",
                "source": "DECamAISched",
                "target": "SISPI",
                "timestamp": self.current_time,
                "transaction_id": str(self.transaction_id),
                "command": "EXPOSE",
            }
            self.transaction_id += 1
        elif msg_type == "TELEMETRY":
            cmd = {
                "type": "TELEMETRY",
                "command": "TELEMETRY",
            }
        else:
            raise ValueError(f"[Client] Unsupported message type: {msg_type}")
        return cmd

    def get_telemetry(self):
        """
        Fetch live telemetry.
        """
        cmd = self._build_base_message("TELEMETRY")
        
        # send a request for telemetry and parse the response
        try:
            response_str = self.scl_client.send_command(json.dumps(cmd))
            if not response_str:
                return {"pointing_ra": None, "pointing_dec": None}

            response = json.loads(response_str)
            telemetry_data = response.get("telemetry", {})
            
            # extract current RA/Dec, falling back to last known values if not reported
            # XXX should check if this is actually reported in response
            ra = telemetry_data.get("ra", self.current_ra)
            dec = telemetry_data.get("dec", self.current_dec)

            return {"pointing_ra": ra, "pointing_dec": dec}
            
        except Exception as e:
            logger.exception(f"[Client] Error fetching telemetry: {e}")
            return {"pointing_ra": None, "pointing_dec": None}

    def check_telemetry_change(self, current_telemetry, last_telemetry):
        """Returns True if the pointing has changed since the last telemetry check."""
        if not last_telemetry:
            return True
        ra_changed = current_telemetry.get("pointing_ra") != last_telemetry.get("pointing_ra")
        dec_changed = current_telemetry.get("pointing_dec") != last_telemetry.get("pointing_dec")
        return ra_changed or dec_changed
    
    def check_exposure_status(self):
        """Return exposure readiness state from control system."""

        # send a request for telemetry
        cmd = self._build_base_message("TELEMETRY")
        try:
            response_str = self.scl_client.send_command(json.dumps(cmd))
            if not response_str:
                return False

            # server provides a bool indicating if it can accept the next EXPOSE command
            response = json.loads(response_str)
            return response.get("readyToExpose", False)
            
        except Exception as e:
            logger.exception(f"[Client] Error checking exposure status: {e}")
            return False

    def submit_observation(self, obs_row, exp_time=None):
        """Submit one observation request to the control system."""
        cmd = self._build_base_message("COMMAND")
        
        # map the desired observation to the command parameters expected by SCLN
        self.current_ra = float(obs_row.get("ra", self.current_ra))
        self.current_dec = float(obs_row.get("dec", self.current_dec))
        cmd["parameters"] = {
            "expTime": str(obs_row.get("expTime", 90)) if exp_time is None else str(exp_time), # XXX examples had this as str, but directions say int
            "expType": "dark", # XXX dark for day-time testing #str(obs_row.get("expType", "object")),
            "propid": str(obs_row.get("propid", "UNKNOWN")) if self.propid is None else str(self.propid),
            "count": int(obs_row.get("count", 1)),
            "filter": "block", # XXX block for day-time testing #str(obs_row.get("filter", "None")),
            "ra": self.current_ra / units.degree,
            "dec": self.current_dec / units.degree,
            "object": str(obs_row.get("field_name", f"pointing_{self.current_time}")),
            "comment": "DO NOT USE", # XXX placeholder for day-time testing
        }

        # send the command and wait for the synchronous response
        logger.info(f"[Client] SUBMIT: RA={cmd['parameters']['ra']}, DEC={cmd['parameters']['dec']}, FILTER={cmd['parameters']['filter']}")
        try:
            response_str = self.scl_client.send_command(json.dumps(cmd))
            response = json.loads(response_str) if response_str else {}
            
            if response.get("status") == "FAILED":
                logger.warning(f"[Client] EXPOSURE FAILED: {response.get('message')}")
                
            return response
            
        except Exception as e:
            logger.exception(f"[Client] Error submitting observation: {e}")
            return None

    def close(self):
        """Clean up the SCL client connection."""
        self.scl_client.close()
        logger.info("[Client] Closed connection to SCLN server.")
