import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


# Abstract Base Class the Model Runner, which generates observing chunks
class ModelRunner(ABC):
    # generate an observing chunk plan
    @abstractmethod
    def generate_chunk(self, telemetry, available_fields, masked_fields, chunk_size):
        pass

class MockModelRunner(ModelRunner):
    import random
    from blancops.math import geometry, units
    from blancops.ephemerides import ephemerides

    def generate_next_observation(self, telemetry, masked_fields):
        current_ra, current_dec = telemetry["pointing_ra"], telemetry["pointing_dec"]
        ra, dec = current_ra, current_dec
        masked_fields = np.asarray(masked_fields)

        # generate a random nearby pointing, ensuring not in excluded regions
        valid = False
        while not valid:
            drad = 10 * self.units.degree
            delta_ra = self.random.uniform(-drad, drad) / np.cos(current_dec)
            delta_dec = self.random.uniform(-drad, drad)
            ra = current_ra + delta_ra
            dec = current_dec + delta_dec

            # check that the new pointing is reasonably far from current pointing
            angsep = self.geometry.angular_separation((ra, dec), (current_ra, current_dec))
            valid_currentangsep = angsep > 1.5 * self.units.degree # 1.5deg minimum step
            if not valid_currentangsep:
                continue

            # check that the new pointing is not nearby any masked fields
            angsep = self.geometry.angular_separation((ra, dec), masked_fields)
            valid_angsep = np.all(angsep > 5 * self.units.degree) # 5deg threshold
            if not valid_angsep:
                continue

            # check that the new pointing is within the Blanco's observable sky
            el = self.ephemerides.equatorial_to_topographic(ra, dec)
            valid_el = el > 30 * self.units.degree # 30deg elevation limit

            valid = valid_currentangsep and valid_angsep and valid_el

        return ra, dec


    def generate_chunk(self, telemetry, available_fields, masked_fields, chunk_size):
        print("[Model] Generating mock observing chunk based on telemetry and field masks...")
        out = []

        # generate mocked fields, randomly distributed around each previous field
        ra, dec = telemetry["pointing_ra"], telemetry["pointing_dec"]
        for i in range(chunk_size):
            ra, dec = self.generate_next_observation(
                telemetry={"pointing_ra": ra, "pointing_dec": dec},
                masked_fields=masked_fields,
            )

            # add to the output dataframe
            out.append({
                'time': i,
                'ra': ra,
                'dec': dec,
                'filter': self.random.choice(['g', 'r', 'i', 'z'])
            })

        return pd.DataFrame(out)

class AIModelRunner(ModelRunner):
    def __init__(self, model_path):
        self.model_path = model_path
        # Placeholder: Load model architecture and weights into memory
        print(f"[Model] Loaded model weights from {model_path} into memory.")

    def generate_chunk(self, telemetry, available_fields, masked_fields, chunk_size):
        print("[Model] Generating state features and running inference...")
        
        # Mocking inference output as a Pandas DataFrame
        mock_data = {
            'field_id': [101, 102, 103, 104, 105][:chunk_size],
            'est_time': ['20:00:00', '20:05:00', '20:10:00', '20:15:00', '20:20:00'][:chunk_size],
            'ra': np.random.uniform(10, 20, chunk_size),
            'dec': np.random.uniform(-50, -40, chunk_size),
            'filter': ['r', 'i', 'z', 'r', 'g'][:chunk_size]
        }
        return pd.DataFrame(mock_data)