import pytest

sktime = pytest.importorskip("sktime")

from sktime.forecasting.naive import NaiveForecaster  # noqa: E402

from pymongopredict import MongoSeries  # noqa: E402


def test_series_split_and_sktime_forecast(observations):
    ts = MongoSeries(
        observations,
        time_field="ts",
        value_field="airTemperature.value",
        freq="h",
        agg="mean",
        valid=lambda v: v < 900,
    )
    assert len(ts) == 200 and ts.series.index.freqstr.lower().startswith("h")
    y_train, y_test = ts.split(test_size=24)
    assert len(y_train) == 176 and len(y_test) == 24
    fc = NaiveForecaster(strategy="last", sp=24).fit(y_train)
    pred = fc.predict(fh=list(range(1, 25)))
    assert len(pred) == 24
    assert (pred.values - y_test.values).__abs__().max() < 1e-6  # daily-periodic synthetic data


def test_series_daily_bucketing(observations):
    ts = MongoSeries(observations, "ts", "airTemperature.value", freq="D", valid=lambda v: v < 900)
    y_train, y_test = ts.split(test_size=0.25)
    assert len(ts) == 9 and len(y_test) == 2
