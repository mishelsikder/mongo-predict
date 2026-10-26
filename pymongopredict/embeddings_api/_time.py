"""Time values and test sizes, shared by the featurizer's date handling and ``MongoSeries``.

Time values may be datetimes, dates, epoch seconds or milliseconds, or date strings; they are
compared as naive UTC.
"""

from __future__ import annotations

import datetime as dt
import numbers

import numpy as np
import pandas as pd


def _is_count(test_size) -> bool:
    """Whether ``test_size`` is a count of documents (or periods) rather than a fraction.

    Parameters
    ----------
    test_size : float or int
        The value given by the user.

    Returns
    -------
    bool
        ``True`` for an integer (Python or numpy, not bool) of at least 1. Floats, including
        ``1.0``, are fractions, as in sklearn.

    Examples
    --------
    >>> _is_count(10), _is_count(0.2), _is_count(True)
    (True, False, False)
    """
    integral = isinstance(test_size, numbers.Integral) and not isinstance(test_size, bool)
    return integral and test_size >= 1


def _is_nan(v) -> bool:
    """Missing: None, NaN, pd.NA, NaT (of any kind); containers are never missing."""
    if v is None:
        return True
    try:
        return np.ndim(v) == 0 and bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _naive_utc(t: dt.datetime) -> dt.datetime:
    """Naive UTC, as pymongo returns stored dates: aware values are converted, not just stripped."""
    return t.astimezone(dt.timezone.utc).replace(tzinfo=None) if t.tzinfo is not None else t


def _as_datetime(v):
    """A naive UTC datetime, or ``None`` when ``v`` is missing or not a time.

    Numbers are epoch seconds, or epoch milliseconds when larger than 1e11 (past the year 5000
    in seconds), as JavaScript and many MongoDB exports store them.
    """
    if _is_nan(v) or isinstance(v, (bool, np.bool_)):
        return None
    if type(v).__name__ == "DatetimeMS":  # pymongo's out-of-range dates (DATETIME_AUTO)
        try:
            return v.as_datetime().replace(tzinfo=None)
        except (ValueError, OverflowError, OSError):
            return None
    if isinstance(v, dt.datetime):  # includes pandas Timestamp
        return _naive_utc(v.to_pydatetime(warn=False) if isinstance(v, pd.Timestamp) else v)
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day)
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        if abs(x) > 1e11:
            x /= 1000.0
        try:
            return dt.datetime.fromtimestamp(x, tz=dt.timezone.utc).replace(tzinfo=None)
        except (ValueError, OverflowError, OSError):
            return None
    if isinstance(v, np.datetime64):
        t = pd.Timestamp(v)
        return None if pd.isna(t) else t.to_pydatetime()
    if isinstance(v, str):
        try:
            return _naive_utc(dt.datetime.fromisoformat(v.replace("Z", "+00:00")))
        except ValueError:
            pass
        try:  # "Jan-2015", "Wed Aug 30 07:12:41 -0700 2017", "03/05/1984" ...
            t = pd.to_datetime(v, errors="coerce", utc=True)
            return None if pd.isna(t) else t.tz_convert(None).to_pydatetime()
        except Exception:  # noqa: BLE001
            return None
    return None
