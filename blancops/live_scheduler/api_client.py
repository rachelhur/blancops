from abc import ABC, abstractmethod


# Abstract Base Class for API Client to interact with the telescope control system
class TelescopeAPI(ABC):
    # poll current telemetry (pointing, weather, etc.) from the telescope control system
    @abstractmethod
    def get_telemetry(self):
        pass

    # poll the status of the current exposure (idle/exposing/done)
    @abstractmethod
    def check_exposure_status(self):
        pass

    # submit an observation request to the telescope control system
    @abstractmethod
    def submit_observation(self, obs_row):
        pass


class MockTelescopeAPI(TelescopeAPI):
    import time
    from blancops.math import units, geometry

    def __init__(self, exposure_duration=90):
        from blancops.ephemerides import ephemerides

        # model internal state to simulate exposure timing
        self.last_exposure_submit_time = -float("inf")
        self.exposure_duration = exposure_duration
        self.slew_time = 0

        # track current pointing to model stepping through observations
        self.current_ra, self.current_dec = self.ephemerides.get_source_ra_dec("zenith")

        print("[API] Initialized mock telescope API connection.")

    def get_telemetry(self):
        # return mock stored pointing
        return {"pointing_ra": self.current_ra, "pointing_dec": self.current_dec}

    def check_exposure_status(self):
        # simulate timing based on last submission time and fixed exposure/slew timing
        delta = self.time.time() - self.last_exposure_submit_time
        return delta > self.slew_time + self.exposure_duration

    def submit_observation(self, obs_row):
        # model slew time based on angular separation between pointings
        angsep = self.geometry.angular_separation(
            (self.current_ra, self.current_dec), (obs_row["ra"], obs_row["dec"])
        )
        self.slew_time = self.geometry.blanco_slew_time(angsep) / self.units.second

        # update internal state to reflect the new observation
        self.last_exposure_submit_time = self.time.time()
        self.current_ra = obs_row["ra"]
        self.current_dec = obs_row["dec"]

        print(
            f"\n[API] ---> SUBMITTING TO QUEUE: Field ID: {obs_row['field_id']} | Filter: {obs_row['filter']}"
        )


class BlancoTelescopeAPI(TelescopeAPI):
    def __init__(self):
        self.connected = True
        print("[API] Initialized connection to telescope control system.")

    def get_telemetry(self):
        # Placeholder: Poll current telemetry
        return {"pointing_ra": 12.5, "pointing_dec": -45.2, "wind_speed": 5.0}

    def check_exposure_status(self):
        # Placeholder: True if idle/done, False if exposing.
        # Mocked to True to simulate immediate completion for testing.
        return True

    def submit_observation(self, obs_row):
        # Placeholder: Issue synchronous command to API
        print(
            f"\n[API] ---> SUBMITTING TO QUEUE: Field ID: {obs_row['field_id']} | Filter: {obs_row['filter']}"
        )
