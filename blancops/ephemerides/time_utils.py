from datetime import datetime, timezone
from dateutil.parser import parse


def utc_now():
    """
    Return the current time in UTC as UNIX timestamp.
    """
    return datetime.now(tz=timezone.utc).timestamp()


def datetime_to_unix(dt, tz=timezone.utc):
    """
    Convert a datetime object to a UNIX timestamp.

    Arguments
    ---------
    dt: datetime
        Datetime object to convert. Should be timezone-aware.
    tz: timezone
        Timezone to use for conversion if dt is timezone-naive. Default is UTC.

    Returns
    -------
    float
        Corresponding UNIX timestamp.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.timestamp()


def unix_to_datetime(ts):
    """
    Convert a UNIX timestamp to a timezone-aware datetime object in UTC.

    Arguments
    ---------
    ts: float
        UNIX timestamp in UTC to convert.

    Returns
    -------
    datetime
        Corresponding timezone-aware datetime object in UTC.
    """
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def unix_to_local_datetime(ts):
    """
    Convert a UNIX timestamp in UTC to a datetime object in the local timezone.

    Arguments
    ---------
    ts: float
        UNIX timestamp in UTC to convert.

    Returns
    -------
    datetime
        Corresponding timezone-aware datetime object in the local timezone.
    """
    return unix_to_datetime(ts).astimezone()


def standardize_time(t):
    """
    Ensures a time variable is in standard format: UNIX timestamp in UTC. Assumes that
    numerical inputs are already in correct format. Parses string inputs using dateutil
    for flexibility. Objects without a clear timezone are assumed to be in UTC.


    Arguments
    ---------
    t: float, str, datetime
        Time variable to standardize.

    Returns
    -------
    float
        UNIX timestamp in UTC.
    """
    # helper function to check if a string can be parsed as a number
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # keep numerical inputs as is, parse strings into numbers if possible else datetime
    if isinstance(t, (int, float)):
        return t
    elif isinstance(t, str):
        if is_number(t):
            return float(t)
        dt = parse(t)
    elif isinstance(t, datetime):
        dt = t
    else:
        raise ValueError("Unsupported time format")

    # assume datetimes without timezone info are in UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()
