"""Demo 2 (time series): the ``sample_weatherdata.data`` collection, 10,000 surface observations
from 5 to 13 March 1984.

Part A: regression with a **time based split**. Predict each observation's air temperature from
the other fields, training on the first 80% of the period and testing on the last 20%, with
gradient boosting on (a) tabular columns, (b) a voyage-4-nano embedding of the whole JSON
document (target removed), (c) both.

Part B: **forecasting with sktime**. Aggregate the hourly mean air temperature over all stations
and forecast the final 24 hours with a few sktime forecasters.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from demos._common import Timer, collection, embedding_kwargs, markdown_table, parser, xgb_estimator
from pymongopredict import DocumentFeaturizer, MongoDataset, MongoSeries

TARGET = "airTemperature.value"
COLUMNS = [
    "position.coordinates.0",
    "position.coordinates.1",
    "elevation",
    "dewPoint.value",
    "pressure.value",
    "wind.speed.rate",
    "wind.direction.angle",
    "visibility.distance.value",
    "skyCondition.ceilingHeight.value",
    "type",
    "dataSource",
    "callLetters",
]
VALID = {TARGET: {"$lt": 900}}  # 999.9 encodes a missing temperature


def part_a(args, coll):
    print("Part A: air temperature regression with a time based split (last 20% of period is test)")
    ds = MongoDataset(
        coll,
        target=TARGET,
        filter=VALID,
        sort=[("ts", 1)],  # in time order: the last 20% are the newest
        limit=args.limit,
        streaming=args.streaming,
        batch_size=args.batch_size,
    )
    ds_train, ds_test = train_test_split(ds, test_size=0.2, shuffle=False)
    y_tr, y_te = ds_train.y, ds_test.y
    common = dict(
        cache_dir=args.cache_dir,
        embedding_model=args.embedding_model,
        embedding_kwargs=embedding_kwargs(args),
    )
    configs = {
        "columns only": dict(columns=COLUMNS, embed_fields=False),
        "embedding of full document": dict(embed_fields=None),
        "columns + embedding": dict(columns=COLUMNS, embed_fields=None),
    }
    base = np.full_like(y_te, y_tr.mean())
    rows = [
        {
            "features": "predict training mean",
            "r2": r2_score(y_te, base),
            "mae": mean_absolute_error(y_te, base),
            "rmse": root_mean_squared_error(y_te, base),
            "n_features": 0,
            "seconds": 0.0,
        }
    ]
    estimators = {"gradient boosting": lambda: HistGradientBoostingRegressor(random_state=0)}
    if xgb_estimator("reg") is not None:  # optional: only when xgboost is installed
        estimators["xgboost"] = lambda: xgb_estimator("reg")
    for (name, feats), (est_name, make_est) in itertools.product(
        configs.items(), estimators.items()
    ):
        prep = DocumentFeaturizer(**common, **feats)
        with Timer() as t:
            reg = make_est().fit(prep.fit_transform(ds_train), y_tr)
            pred = reg.predict(prep.transform(ds_test))
        name = f"{name} ({est_name})"
        rows.append(
            {
                "features": name,
                "r2": r2_score(y_te, pred),
                "mae": mean_absolute_error(y_te, pred),
                "rmse": root_mean_squared_error(y_te, pred),
                "n_features": len(prep.get_feature_names_out()),
                "seconds": t.seconds,
            }
        )
        print(f"{name}: r2={rows[-1]['r2']:.3f}", flush=True)
    ts_tr, ts_te = ds_train.field_values("ts"), ds_test.field_values("ts")
    print(
        f"\ntrain: {len(ds_train)} obs, {min(ts_tr):%Y-%m-%d %H:%M} to "
        f"{max(ts_tr):%Y-%m-%d %H:%M}; test: {len(ds_test)} obs, "
        f"{min(ts_te):%Y-%m-%d %H:%M} to {max(ts_te):%Y-%m-%d %H:%M}\n"
    )
    print(markdown_table(rows))


def part_b(args, coll):
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.trend import PolynomialTrendForecaster

    print(
        "\nPart B: forecasting the hourly mean air temperature (all stations) "
        "for the last 24 hours with sktime"
    )
    series = MongoSeries(
        coll,
        time_field="ts",
        value_field=TARGET,
        freq="h",
        agg="mean",
        valid=lambda v: v < 900,
        limit=args.limit,
    )
    y_train, y_test = series.split(test_size=24)
    fh = np.arange(1, len(y_test) + 1)
    forecasters = {
        "naive last value": NaiveForecaster(strategy="last"),
        "naive mean of last 24h": NaiveForecaster(strategy="mean", window_length=24),
        "seasonal naive (sp=24)": NaiveForecaster(strategy="last", sp=24),
        "Holt-Winters additive (sp=24)": ExponentialSmoothing(trend="add", seasonal="add", sp=24),
        "linear trend": PolynomialTrendForecaster(degree=1),
    }
    rows = []
    for name, f in forecasters.items():
        pred = f.fit(y_train).predict(fh=fh)
        rows.append(
            {
                "forecaster": name,
                "mae": mean_absolute_error(y_test, pred),
                "rmse": root_mean_squared_error(y_test, pred),
            }
        )
    s = series.series
    print(
        f"series: {len(s)} hourly points from {s.index[0]} to {s.index[-1]}, "
        f"train {len(y_train)} / test {len(y_test)}\n"
    )
    print(markdown_table(rows))


def main():
    args = parser(__doc__).parse_args()
    coll = collection(args.uri, "sample_weatherdata", "data")
    part_a(args, coll)
    part_b(args, coll)


if __name__ == "__main__":
    main()
