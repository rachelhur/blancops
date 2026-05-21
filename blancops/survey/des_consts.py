from __future__ import annotations

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



_DES_SUN_EL_LIMIT = -10.5 # min sun el limit seen in processed data
_VALID_TEFF_THRESHOLD = 0.3