"""Time series: ``MongoSeries`` aggregates a series out of a collection and ``MongoSeries.split``
cuts it in time order; the result is a plain ``pd.Series`` for any sktime forecaster.

:class:`MongoSeries` reads only a time field and a value field from the documents (through a
``find`` projection), buckets the values into regular periods with pandas, and returns a
``pd.Series`` with a ``PeriodIndex``. Times may be datetimes, dates, epoch seconds or
milliseconds, or date strings, compared as naive UTC.

The private helpers translate a pandas frequency alias between the two forms pandas wants:
the resampling alias (``"ME"``, ``"QE"``) and the period alias (``"M"``, ``"Q-DEC"``).

Examples
--------
>>> from sktime.forecasting.naive import NaiveForecaster  # doctest: +SKIP
>>> ms = MongoSeries(weather, "ts", "airTemperature.value", freq="D", agg="mean",
...                  valid=lambda v: v != 999.9)  # doctest: +SKIP
>>> train, test = ms.split(test_size=14)  # doctest: +SKIP
>>> NaiveForecaster().fit(train).predict(fh=range(1, 15))  # doctest: +SKIP
"""

from __future__ import annotations

import math
import re

import pandas as pd

from ._time import _as_datetime, _is_count
from .documents import _get_path, _projection_for, _scalar

# period-style aliases that pandas 3 no longer accepts for resampling
#: a larger series is refused (see MongoSeries.load)
MAX_PERIODS = 1_000_000

_RESAMPLE_ALIASES = {
    "M": "ME",
    "Q": "QE",
    "Y": "YE",
    "A": "YE",
    "BM": "BME",
    "BQ": "BQE",
    "BA": "BYE",
    "BY": "BYE",
}
_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _parse_freq(freq: str):
    """Split an alias such as ``"2QS-FEB"`` into ``("2", "QS", "FEB")`` (multiple, base, anchor).

    The base and anchor are upper-cased; ``None`` when the alias has another shape.
    """
    m = re.fullmatch(r"(\d*)([A-Za-z]+)(?:-([A-Za-z]{3}))?", freq)
    if not m:
        return None
    return m.group(1), m.group(2).upper(), (m.group(3) or "").upper()


def _resample_freq(freq: str) -> str:
    """The alias pandas 3 resamples with (period-style M / Q / Y become ME / QE / YE).

    The multiple and anchor are kept (``"2Q-JAN"`` -> ``"2QE-JAN"``); other aliases pass through.
    """
    parsed = _parse_freq(freq)
    if not parsed or parsed[1] not in _RESAMPLE_ALIASES:
        return freq
    n, base, anchor = parsed
    return n + _RESAMPLE_ALIASES[base] + (f"-{anchor}" if anchor else "")


def _period_freq(freq: str) -> str | None:
    """The period frequency of resampled bins (with the multiple: 2MS -> 2M), inferred from the
    index when not a month / quarter / year alias; anchored starts end their periods the month
    before (QS-FEB -> Q-JAN)."""
    parsed = _parse_freq(freq)
    if not parsed:
        return None
    n, base, anchor = parsed
    base = (
        base[1:] if base.startswith("B") and base not in ("B", "BH") else base
    )  # business variants (BMS, BQE ...) give the same periods as their calendar base
    if base in ("M", "ME", "MS"):
        return f"{n}M"
    if base in ("Q", "QE", "QS", "Y", "YE", "YS", "A", "AS"):
        kind = "Q" if base.startswith("Q") else "Y"
        if base.endswith("S"):  # periods start at the anchor month: they end the month before
            end = _MONTHS[_MONTHS.index(anchor or "JAN") - 1]
        else:
            end = anchor or "DEC"
        return f"{n}{kind}-{end}"
    return None


class MongoSeries:
    """A univariate time series aggregated from the documents of a collection.

    Only the two fields are read, streamed through the cursor. ``series`` is a ``pd.Series``
    with a ``PeriodIndex``, ready for sktime. The collection is read lazily, on the first
    access to ``series``, ``len()`` or :meth:`split`, and the result is cached.

    Parameters
    ----------
    collection : pymongo.collection.Collection
        The collection to read.
    time_field : str
        Dotted path of the timestamp field. Accepted values: ``datetime`` / ``date`` /
        ``pd.Timestamp``, ISO 8601 strings, and numbers as epoch seconds (or epoch milliseconds
        when above 1e11). Timezone-aware times are converted to naive UTC. Documents whose time
        cannot be parsed are skipped.
    value_field : str
        Dotted path of the numeric field to aggregate. ``Decimal128`` values are read as floats;
        values that are missing, ``None`` or not convertible to ``float`` are skipped.
    freq : str, default "D"
        pandas offset alias used to bucket timestamps (``"h"``, ``"D"``, ``"W"``, ``"MS"`` ...),
        with an optional multiple and anchor (``"2MS"``, ``"QS-FEB"``). The period-style
        aliases ``"M"``, ``"Q"``, ``"Y"`` and ``"A"`` are accepted and resampled as ``"ME"``,
        ``"QE"`` and ``"YE"``, which pandas 3 requires. Month, quarter and year aliases (also
        business ones) map to the matching periods, keeping the multiple (``"2MS"`` gives
        ``"2M"`` periods) and ending start-anchored periods the month before the anchor
        (``"QS-FEB"`` gives ``"Q-JAN"``). For other aliases the period frequency is inferred
        from the resampled index.
    agg : str, default "mean"
        Aggregation within a bucket, any name pandas ``Resampler.agg`` accepts (``"mean"``,
        ``"sum"``, ``"count"``, ``"median"``, ``"max"`` ...).
    fill : {"interpolate", "ffill"}, number or None, default "interpolate"
        How to fill empty buckets: ``"interpolate"`` interpolates linearly and extends the
        first and last values to the ends; ``"ffill"`` carries the last value forward (and the
        first value backward for leading gaps); a number fills with that value; ``None`` leaves
        ``NaN``. Aggregations such as ``"count"`` and ``"sum"`` give 0, not ``NaN``, for empty
        buckets, so there is nothing to fill.
    filter : dict or None, default None
        ``find`` filter; ``None`` reads every document.
    valid : callable or None, default None
        ``valid(value) -> bool`` to drop sentinel values (e.g. ``999.9`` in the weather sample
        data). It sees the value before conversion to ``float``.
    limit : int or None, default None
        Read at most this many documents (in natural order); ``None`` or ``0`` reads all.

    Attributes
    ----------
    series : pandas.Series
        The aggregated series, named after ``value_field``.

    Examples
    --------
    >>> ms = MongoSeries(coll, "ts", "airTemperature.value", freq="D")  # doctest: +SKIP
    >>> train, test = ms.split(0.2)  # doctest: +SKIP
    """

    def __init__(
        self,
        collection,
        time_field: str,
        value_field: str,
        freq: str = "D",
        agg: str = "mean",
        fill="interpolate",
        filter: dict | None = None,
        valid=None,
        limit: int | None = None,
    ):
        self.collection = collection
        self.time_field = time_field
        self.value_field = value_field
        self.freq = freq
        self.agg = agg
        self.fill = fill
        self.filter = filter
        self.valid = valid
        self.limit = limit
        self._series: pd.Series | None = None

    def load(self, force: bool = False):
        """Read the collection and build the series (once, unless ``force``).

        Parameters
        ----------
        force : bool, default False
            Re-read the collection even if the series is already cached, e.g. after new
            documents were inserted.

        Returns
        -------
        MongoSeries
            ``self``, so that ``ms.load().series`` chains.

        Raises
        ------
        ValueError
            If fewer than three documents have a usable (time, value) pair, or fewer than three
            periods remain after resampling.
        """
        if self._series is not None and not force:
            return self
        cursor = self.collection.find(
            self.filter or {}, _projection_for([self.time_field, self.value_field])
        )
        if self.limit:
            cursor = cursor.limit(int(self.limit))
        rows = []
        for d in cursor:
            # times as in time splits: naive UTC, epoch seconds / milliseconds for numbers
            t, v = (
                _as_datetime(_get_path(d, self.time_field)),
                _scalar(_get_path(d, self.value_field)),
            )
            if t is None or v is None or (self.valid is not None and not self.valid(v)):
                continue
            if isinstance(v, bool):  # not a measurement
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):  # a value that is not a number ("n/a", a list)
                continue
            if not math.isfinite(v):
                continue
            rows.append((pd.Timestamp(t), v))
        if len(rows) < 3:
            raise ValueError("fewer than three usable (time, value) pairs")
        df = pd.DataFrame(rows, columns=["t", "v"]).set_index("t").sort_index()
        self._check_span(df.index[0], df.index[-1])
        s = df["v"].resample(_resample_freq(self.freq)).agg(self.agg)
        if self.fill == "interpolate":
            s = s.interpolate(limit_direction="both")
        elif self.fill == "ffill":
            s = s.ffill().bfill()
        elif self.fill is not None:
            s = s.fillna(self.fill)
        s = s.astype(float)
        # sktime forecasters want a regular PeriodIndex rather than bin-edge timestamps
        try:
            s.index = s.index.to_period(_period_freq(self.freq))
        except ValueError as e:  # e.g. semi-month bins ("SME", "SMS") have no period frequency
            raise ValueError(
                f"freq={self.freq!r} has no matching pandas period, which sktime needs; use a "
                "frequency such as 'h', 'D', 'W', 'MS', 'QS' or 'YS'"
            ) from e
        s.name = self.value_field
        if len(s) < 3:
            raise ValueError(f"fewer than three {self.freq!r} periods after resampling")
        self._series = s
        return self

    @property
    def series(self) -> pd.Series:
        """The aggregated series (float values, ``PeriodIndex``); loads it on first access."""
        return self.load()._series

    def _check_span(self, first, last) -> None:
        """Refuse a series of more than ``MAX_PERIODS`` periods before resampling allocates it
        (one document dated 9999-12-31 would make billions of hourly periods)."""
        try:
            step = pd.tseries.frequencies.to_offset(_resample_freq(self.freq)).nanos
        except ValueError:  # calendar frequencies (months, quarters, years): never that many
            return
        # (in Python datetimes: a span of thousands of years overflows pandas' nanoseconds)
        span = (last.to_pydatetime() - first.to_pydatetime()).total_seconds()
        n = int(span * 1e9 // max(step, 1)) + 1
        if n > MAX_PERIODS:
            raise ValueError(
                f"{self.time_field!r} spans {first} to {last}: {n:,} {self.freq!r} periods "
                f"(more than {MAX_PERIODS:,}); filter out the outlying dates or use a coarser freq"
            )

    def __len__(self) -> int:
        """Number of periods in the series (loads it if needed)."""
        return len(self.series)

    def split(self, test_size: float | int = 0.2) -> tuple[pd.Series, pd.Series]:
        """Trailing ``test_size`` periods (int) or fraction (float) become the test series.

        Parameters
        ----------
        test_size : float or int, default 0.2
            An integer of at least 1 is a number of periods; anything else is a fraction of the
            series length (rounded). The test size is clipped so that the test series has at
            least one period and the training series at least two.

        Returns
        -------
        tuple of pandas.Series
            ``(train, test)``, contiguous and in time order.

        Examples
        --------
        >>> train, test = ms.split(test_size=12)  # last 12 periods  # doctest: +SKIP
        """
        s = self.series
        n_test = int(test_size) if _is_count(test_size) else int(round(len(s) * test_size))
        n_test = min(max(int(n_test), 1), len(s) - 2)
        return s.iloc[:-n_test], s.iloc[-n_test:]

    def __repr__(self) -> str:
        """Short description; ``n`` is ``?`` until the series is loaded (repr never reads)."""
        n = len(self._series) if self._series is not None else "?"
        return (
            f"MongoSeries({self.value_field!r} by {self.time_field!r}, freq={self.freq!r}, n={n})"
        )
