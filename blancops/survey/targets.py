from __future__ import annotations

from typing import Literal

import numpy as np
import healpy as hp
import pandas as pd

from blancops.survey.footprint import galactic_plane_mask
from blancops.survey.visibility import visible_fields


def build_candidate_fields(
    tile_radec_rad: np.ndarray,                  # (n, 2) [ra, dec] radians
    nside: int,
    g_lat_lim: float,
    window_utc: tuple[str, str],
    *,
    airmass_limit: float = None,
    require: Literal["any", "all"] = "any",
) -> pd.DataFrame:
    """
    Return candidate fields surviving galactic-plane exclusion AND window
    visibility. Output schema matches LookupTables.fields: columns
    name, ra, dec (radians), plus frac_observable. Index is contiguous 0..N-1.
    """
    ra = tile_radec_rad[:, 0]
    dec = tile_radec_rad[:, 1]

    exclusion = galactic_plane_mask(nside, g_lat_lim)     # (npix,)
    pix = hp.ang2pix(nside, np.pi / 2 - dec, ra)
    in_exclusion = exclusion[pix]

    vis_mask, frac = visible_fields(
        tile_radec_rad, window_utc, airmass_limit=airmass_limit, require=require
    )

    keep = (~in_exclusion) & vis_mask
    n_keep = int(keep.sum())
    return pd.DataFrame(
        {
            "name": [f"field_{i}" for i in range(n_keep)],
            "ra": ra[keep],
            "dec": dec[keep],
            "frac_observable": frac[keep],
        }
    ).reset_index(drop=True)


def to_lookup_fields_df(
    fields: pd.DataFrame,
    propid: str,
    filt: str,
    count: int = 1,
    exptime: float = 90.0,
    *,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> pd.DataFrame:
    """Assemble a per-field DataFrame for LookupTables.build_lookups_from_fields.

    RA/Dec are read in degrees, wrapped to [0, 360), and converted to radians
    (the units the lookup builder requires). filt is one filter per character
    ("z" -> z only, "zg" -> z and g); each field is emitted once per filter,
    sharing the same count and exptime, and tagged with propid.
    """
    ra_deg = fields[ra_col].to_numpy(dtype=float)
    ra_deg = np.where(ra_deg < 0, ra_deg + 360.0, ra_deg)
    dec_deg = fields[dec_col].to_numpy(dtype=float)
    filters = list(filt)
    n = len(ra_deg)
    return pd.DataFrame(
        {
            "RA": np.tile(np.radians(ra_deg), len(filters)),
            "DEC": np.tile(np.radians(dec_deg), len(filters)),
            "filter": np.repeat(filters, n),
            "count": count,
            "exptime": exptime,
            "propid": propid,
        }
    )

def select_in_path(vertices, ra, dec, wrap=180., radius=0.0):
    """
    Checks if a set of (ra, dec) points (in degrees) fall within a given polygon.

    A positive radius (degrees) grows the polygon outward by that margin. matplotlib
    only grows a counter-clockwise polygon, so the vertices are reoriented to CCW
    first, making the radius sign consistent regardless of input winding.
    """
    import matplotlib
    # Copy coordinates to avoid modifying the original arrays
    ra, dec = np.copy(ra), np.copy(dec)

    # Wrap Right Ascension coordinates
    ra -= 360 * (ra > wrap)

    # Reorient to CCW (positive signed area) so a positive radius grows outward
    verts = np.asarray(vertices, dtype=float)
    vx, vy = verts[:, 0], verts[:, 1]
    signed_area = 0.5 * np.sum(vx * np.roll(vy, -1) - np.roll(vx, -1) * vy)
    if signed_area < 0:
        verts = verts[::-1]

    # Create the matplotlib path from the provided vertices
    path = matplotlib.path.Path(verts)

    # Format the test points into an (M, 2) array
    points = np.vstack([ra, dec]).T
    
    # Check which points are inside the path
    sel = path.contains_points(points, radius=radius)
    
    return sel