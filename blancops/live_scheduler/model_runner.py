"""Model-runner interfaces for live scheduler chunk generation.

This module defines the contract used by the scheduler to request a candidate chunk of
observations and provides two implementations:

- MockModelRunner: stochastic sky-walk generator for demos and integration tests.
- AIModelRunner: placeholder for the production ML-backed model.
"""

import random
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from blancops.math import geometry, units
from blancops.ephemerides import ephemerides


class ModelRunner(ABC):
    """
    Abstract interface for observation-chunk generation.

    Implementations are expected to return a pandas DataFrame with one row per proposed
    observation. Downstream components currently expect columns similar to "time", "ra",
    "dec", and "filter".
    """

    @abstractmethod
    def generate_chunk(self, telemetry, available_fields, masked_fields, chunk_size):
        """
        Generate a chunk of proposed observations.

        Arguments
        ---------
        telemetry: dict
            Current telescope/sky state dictionary. The mock runner expects at least
            "pointing_ra" and "pointing_dec".
        available_fields: list
            Candidate field set available for scheduling.
        masked_fields: list
            Fields to avoid in this proposal cycle.
        chunk_size: int
            Number of sequential observations to propose.

        Returns
        -------
            pandas.DataFrame: Proposed observation chunk.
        """

        pass


class MockModelRunner(ModelRunner):
    """Randomized mock implementation used for development and dry runs."""

    def generate_next_observation(self, telemetry, masked_fields):
        """
        Sample a valid next pointing near the current one.

        The sample is accepted only if it:
        - is at least 1.5 deg from the current pointing,
        - is at least 5 deg from any masked field,
        - is above 30 deg elevation.
        """

        current_ra, current_dec = telemetry["pointing_ra"], telemetry["pointing_dec"]
        ra, dec = current_ra, current_dec
        masked_fields = np.asarray(masked_fields)

        # rejection-sample nearby pointings until one passes all validity checks
        valid = False
        while not valid:
            drad = 10 * units.deg
            delta_ra = random.uniform(-drad, drad) / np.cos(current_dec)
            delta_dec = random.uniform(-drad, drad)
            ra = (current_ra + delta_ra) % (360 * units.deg)
            dec = np.clip(current_dec + delta_dec, -90 * units.deg, 90 * units.deg)

            # Ensure minimum step size from current pointing.
            angsep = geometry.angular_separation((ra, dec), (current_ra, current_dec))
            valid_currentangsep = angsep > 1.5 * units.deg  # 1.5deg minimum step
            if not valid_currentangsep:
                continue

            # keep away from user/system-masked fields
            if len(masked_fields) > 0:
                angsep = geometry.angular_separation((ra, dec), masked_fields)
                valid_angsep = np.all(angsep > 5 * units.deg)  # 5deg threshold
            else:
                valid_angsep = True
            if not valid_angsep:
                continue

            # enforce telescope visibility via elevation floor
            az, el = ephemerides.equatorial_to_topographic(ra, dec)
            valid_el = el > 30 * units.deg  # 30deg elevation limit

            valid = valid_currentangsep and valid_angsep and valid_el

        return ra, dec

    def generate_chunk(self, telemetry, available_fields, masked_fields, chunk_size):
        """
        Generate a mock chunk as a short random walk in sky coordinates.

        Note:
            available_fields is currently unused but included to preserve interface
            compatibility with other production model runners.
        """

        print(
            "[Model] Generating mock observing chunk based on telemetry and field masks..."
        )
        out = []

        # start from the current telescope pointing and walk forward
        ra, dec = telemetry["pointing_ra"], telemetry["pointing_dec"]
        for i in range(chunk_size):
            ra, dec = self.generate_next_observation(
                telemetry={"pointing_ra": ra, "pointing_dec": dec},
                masked_fields=masked_fields,
            )

            # keep output schema aligned with scheduler expectations
            out.append(
                {
                    "time": i,
                    "ra": ra,
                    "dec": dec,
                    "filter": random.choice(["g", "r", "i", "z", "Y"]),
                }
            )

        return pd.DataFrame(out)


class AIModelRunner(ModelRunner):
    """Placeholder ML-backed runner for production inference."""

    def __init__(self, model_path):
        """
        Initialize model resources.

        Arguments
        ---------
        model_path: str
            Filesystem path to serialized model artifacts.
        """

        self.model_path = model_path
        # XXX Placeholder: Load model architecture and weights into memory
        print(f"[Model] Loaded model weights from {model_path} into memory.")

    def generate_chunk(self, telemetry, available_fields, masked_fields, chunk_size):
        """
        Generate an observation chunk from model inference.

        This implementation is a placeholder and currently returns an empty
        DataFrame until model feature generation and inference is wired in.
        """

        print("[Model] Generating state features and running inference...")
        # XXX Placeholder: generate a chunk of observations using the loaded model
        return pd.DataFrame()
