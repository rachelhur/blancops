import fitsio
import numpy as np
import pandas as pd

from astropy.time import Time
import pandas as pd
from astropy.coordinates import EarthLocation
import astropy.units as u


def preprocess_fits(fits_path):
    df = fits_to_df(fits_path)
    df = df.pipe(_replace_with_pd_dt)\
            .pipe(_drop_nan_dts)\
            .pipe(_add_timestamp)\
            .pipe(_add_night)
    return df


def fits_to_df(fits_path):
    d = fitsio.read(fits_path)
    df = pd.DataFrame(d.astype(d.dtype.newbyteorder('='))) # Big-endian/little-endian error
    return df

def _replace_with_pd_dt(df):
    df['datetime'] = pd.to_datetime(
        df['datetime'], 
        format='%Y-%m-%d %H:%M:%S', 
        utc=True, 
        errors='coerce'
    )
    return df

def _drop_nan_dts(df):
    df = df.dropna(subset=['datetime'])
    return df

def _add_timestamp(df):
    t_array = Time(df['datetime'].dt.tz_localize(None).values, scale='utc')
    # .assign() creates and returns a new df with the added column
    return df.assign(timestamp=t_array.unix.astype(np.int64))
def _add_night(df):
    return df.assign(night=(df['datetime'] - pd.Timedelta(hours=12)).dt.date)