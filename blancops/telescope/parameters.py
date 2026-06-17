from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SlewModel:
    """
    Kinematic slew model for a single axis (azimuth or altitude).

    Assumes constant acceleration to max_speed, cruise at max_speed, then
    symmetric deceleration.  For short moves that never reach max_speed the
    formula collapses to the pure-acceleration case.

    All values in degrees and seconds.
    """

    max_speed: float    # deg / s  — peak angular velocity
    acceleration: float # deg / s² — constant accel / decel magnitude

    def slew_time(self, distance: float) -> float:
        """
        Seconds required to move `distance` degrees on this axis.
        distance must be non-negative (pass abs(delta) at the call site).
        """
        if distance <= 0.0:
            return 0.0
        # Distance covered while accelerating to full speed (and decelerating back)
        accel_dist = self.max_speed ** 2 / self.acceleration
        if distance <= accel_dist:
            # Triangular profile — never reaches max_speed
            return 2.0 * math.sqrt(distance / self.acceleration)
        # Trapezoidal profile — reach max_speed and cruise
        return distance / self.max_speed + self.max_speed / self.acceleration


@dataclass(frozen=True)
class TelescopeParameters:
    """
    Hardware capabilities and fixed timing constants for one telescope +
    instrument combination.

    Timing convention (per visit):
        total_time = exposure + readout + shutter_overhead [+ filter_change]

    All times in seconds, all angles in degrees.
    """

    # ------------------------------------------------------------------ #
    # Slew                                                                 #
    # ------------------------------------------------------------------ #
    az_slew: SlewModel
    alt_slew: SlewModel

    # ------------------------------------------------------------------ #
    # Instrument timing                                                    #
    # ------------------------------------------------------------------ #
    readout_time: float         # seconds — detector readout after each exposure
    filter_change_time: float   # seconds — time to rotate filter wheel / changer
    shutter_overhead: float     # seconds — open + close per exposure

    # ------------------------------------------------------------------ #
    # Field of view                                                        #
    # ------------------------------------------------------------------ #
    fov_deg: float              # effective diameter of the focal plane, degrees

    # ------------------------------------------------------------------ #
    # Visit duration bounds                                                #
    # ------------------------------------------------------------------ #
    min_visit_duration: float   # seconds — shortest scientifically useful exposure
    max_visit_duration: float   # seconds — scheduler ceiling (not a hardware limit)

    # ------------------------------------------------------------------ #
    # Filter complement                                                    #
    # ------------------------------------------------------------------ #
    filters: tuple[str, ...]    # ordered tuple of available filter names

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def fov_sq_deg(self) -> float:
        """Solid angle of the focal plane in square degrees (circular aperture)."""
        return math.pi * (self.fov_deg / 2.0) ** 2

    # ------------------------------------------------------------------ #
    # Slew time                                                            #
    # ------------------------------------------------------------------ #

    def slew_time(self, daz: float, dalt: float) -> float:
        """
        Total slew time for a simultaneous az + alt move.
        Both axes move at the same time; the total is the slower axis.
        """
        return max(
            self.az_slew.slew_time(abs(daz)),
            self.alt_slew.slew_time(abs(dalt)),
        )

    # ------------------------------------------------------------------ #
    # Per-visit overhead                                                   #
    # ------------------------------------------------------------------ #

    def visit_overhead(self, filter_change: bool = False) -> float:
        """
        Fixed per-visit overhead in seconds (readout + shutter, optionally
        including a filter change).  Does not include slew time.
        """
        total = self.readout_time + self.shutter_overhead
        if filter_change:
            total += self.filter_change_time
        return total

    def __repr__(self) -> str:
        return (
            f"TelescopeParameters("
            f"fov={self.fov_deg}°, "
            f"filters={self.filters}, "
            f"readout={self.readout_time}s)"
        )
