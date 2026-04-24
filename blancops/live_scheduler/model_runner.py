"""Model-runner interfaces for live scheduler chunk generation.

This module defines the contract used by the scheduler to request a candidate chunk of
observations and provides two implementations:

- MockModelRunner: stochastic sky-walk generator for demos and integration tests.
- AIModelRunner: placeholder for the production ML-backed model.
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from blancops.data.constants import IDX2FILTER
from blancops.ephemerides.time_utils import utc_now
from blancops.math import geometry, units
from blancops.ephemerides import ephemerides
from blancops.data.lookup import LookupTables
from blancops.ephemerides.ephemerides import HealpixGrid
from blancops.live_scheduler.inference.helpers import build_env
from blancops.rl.agent_factory import AgentFactory


class ModelRunner(ABC):
    """
    Abstract interface for observation-chunk generation.

    Implementations are expected to return a pandas DataFrame with one row per proposed
    observation. Downstream components currently expect columns similar to "time", "ra",
    "dec", and "filter".
    """

    @abstractmethod
    def __init__(self, chunk_size):
        """
        Initialize the model runner with any necessary setup, such as loading model
        weights or setting up internal state.

        Arguments
        ---------
        chunk_size: int
            Number of sequential observations to propose in each generated chunk.
        """

        pass

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

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.current_field_id = 0

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
                angsep = geometry.angular_separation(
                    (ra, dec), masked_fields[["ra", "dec"]].values.T
                )
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

    def generate_chunk(self, telemetry, available_fields, masked_fields, chunk_size=None):
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
        for i in range(chunk_size or self.chunk_size):
            ra, dec = self.generate_next_observation(
                telemetry={"pointing_ra": ra, "pointing_dec": dec},
                masked_fields=masked_fields,
            )

            # keep output schema aligned with scheduler expectations
            out.append(
                {
                    "field_id": self.current_field_id,
                    "time": i,
                    "ra": ra,
                    "dec": dec,
                    "filter": random.choice(["g", "r", "i", "z", "Y"]),
                }
            )
            self.current_field_id += 1

        return pd.DataFrame(out)


class AIModelRunner(ModelRunner):
    def __init__(self, model_path_or_alias: str, field_lookup_dir: Path, fields_path: Path = None, device: str = "cpu", 
                 field_choice_method: str = "interp"):
        self.device = device
        self.TESTING_MODE = True # REMOVE WHEN ENVIRONMENT IS FULLY FUNCTIONING
        
        # Fields and Lookups
        self.fields_dir = Path(field_lookup_dir)
        self.lookups = self._get_lookups(fields_path, self.fields_dir)
        
        # Agent and Model
        factory = AgentFactory() # Defaults to WORKSPACE / "deployable_models"
        self.agent, self.cfg, norm_stats = factory.build_agent(
            model_path_or_alias=model_path_or_alias,
            lookups=self.lookups,
            field_choice_method=field_choice_method,
            device=self.device
        )
        
        self.model_dir = factory.resolve_model_dir(model_path_or_alias)
        self.env = build_env(self.cfg, self.model_dir, self.lookups)
        self.hpGrid = HealpixGrid(nside=self.cfg.data.nside, is_azel="azel" in self.cfg.data.action_space)
        print(f"[Model] Loaded model weights from {self.model_dir} into memory.")

    @staticmethod
    def _get_lookups(fields_path, fields_dir):
        if fields_path:
            lookups = LookupTables.generate_lookups_from_fields(fields_path=fields_path, outdir=fields_dir, write_to_disk=True)
        else:
            lookups = LookupTables.load_from_dir(fields_dir, is_training=False, is_historic=False, construct_if_missing=True)
        return lookups
        
    def generate_chunk(self, telemetry, available_fields, masked_fields, chunk_size, new_fields=None, new_lookup_dir=None) -> pd.DataFrame:
        
        if self.TESTING_MODE:
            obs, info = self.env.get_obs(), self.env.get_info()
            # ra, dec = telemetry["pointing_ra"], telemetry["pointing_dec"]
            
            # self.env.step()
            raise NotImplementedError
        else:
            # UPDATE TELEMETRY/FIELD LOOKUPS
            telemetry = self.process_telemetry(telemetry)
            self.lookups = self.update_lookups(new_fields, new_dir=new_lookup_dir)
            
            # UPDATE ENV AND GET CURRENT STATE/OBS
            self.env.sync_to_telemetry(telemetry)   # will sync up with (1) telemetry time
                                                                    # (2) current ra, dec 
                                                                    # (3) its counters for (field, filter) for *successful* exposures that have occured since last 
                                                                    # chunk generation
                                                    # note to self, should reset environment counters back to "original state" at the end of policy rollout 
            obs, info = self.env.get_obs(), self.env.get_info()
            
            # GENERATE SCHEDULE
            print("[Model] Generating state features and running inference...")
            proposed_schedule = self.rollout_policy(init_obs=obs, init_info=info, chunk_size=chunk_size)
            return proposed_schedule

    def update_lookups(self, new_fields_path, new_dir=None):
        if not new_fields_path: 
            return self.lookups
        
        new_lookups = LookupTables.generate_lookups_from_fields(fields_path=new_fields_path, write_to_disk=False)
        self.lookups = self.lookups.merge(new_lookups, new_dir=new_dir) # writes to disk if new_dir is not None
        return self.lookups
    
    def rollout_policy(self, init_obs: dict, init_info: dict, chunk_size: int) -> pd.DataFrame:
        proposed_schedule = {'bin_idx': [],
                    'field_id': [],
                    'filter': [],
                    'timestamp': [],
                    'ra': [],
                    'dec': [],
                    }
        
        obs, info = init_obs, init_info
        for i in range(chunk_size):
            bin_idx, filter_idx, field_id = self.agent.choose_bin_filter_field(obs, info, self.hpGrid)
            actions = {'bin': np.int32(bin_idx), 'field_id': np.int32(field_id), 'filter_idx': np.int32(filter_idx)}
            
            proposed_schedule['bin_idx'].append(bin_idx)
            proposed_schedule['field_id'].append(field_id)
            proposed_schedule['filter'].append(IDX2FILTER[filter_idx])
            proposed_schedule['timestamp'].append(info.get('timestamp'))
            
            ra, dec = self.lookups.fid2radec[field_id]
            proposed_schedule['ra'].append(ra)
            proposed_schedule['dec'].append(dec)
            
            obs, reward, terminated, truncated, info = self.env.step(actions)
            if terminated or truncated: #ie, end of night - orchestrator default stops this, but doesn't hurt to have extra check here
                break
        
        for key, val in proposed_schedule.items():
            if key in ['bin_idx', 'field_id', 'timestamp']:
                dtype = np.int64
            elif key in ['ra', 'dec']:
                dtype = np.float64
            proposed_schedule[key] = np.array(val, dtype=dtype)
        
        self.env.reset_to_original_state()
            
        return pd.DataFrame(proposed_schedule)