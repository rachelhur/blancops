from dataclasses import dataclass, replace

@dataclass(frozen=True)
class SurveyProfile:
    name: str
    sun_el_limit: float
    valid_teff_threshold: float
    
DES = SurveyProfile(
    name="des",
    sun_el_limit=-10.5,
    valid_teff_threshold=0.3
)