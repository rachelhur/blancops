from datetime import datetime, timezone


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
