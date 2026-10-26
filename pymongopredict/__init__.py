"""pymongopredict: predictive models on MongoDB collections.

Predict a field of your documents (a class or a number) from the other fields and from
embeddings of the document text, and use the predictions in MongoDB.

The package has two APIs over the same featurizer:

* :mod:`pymongopredict.predictions_api`: a ``$predict`` aggregation stage, modelled on Atlas
  ``$vectorSearch``, backed by prediction models stored in the database. Define a model with
  ``create_prediction_model(PredictionModel(definition, name))``, then use
  ``{"$predict": {"model": name, ...}}`` as the first stage of an ordinary pipeline and read
  the predictions with ``{"$meta": "prediction"}``. No server changes are needed.
* :mod:`pymongopredict.embeddings_api`: the scikit-learn workflow underneath it
  (``MongoDataset``, ``DocumentFeaturizer``, ``MongoSeries``; splits come from scikit-learn) for fitting
  any sklearn estimator or sktime forecaster on columns plus document embeddings.

Everything in ``embeddings_api.__all__`` and ``predictions_api.__all__`` is importable from
here.

The README (``README.md``) introduces the package with an example on ``sample_mflix.movies``. These pages are the
reference: :mod:`pymongopredict.predictions_api` and :mod:`pymongopredict.embeddings_api`.

## Install

The package is not on PyPI yet; install it from GitHub (Python 3.10 or later):

    pip install "pymongopredict @ git+https://github.com/voyage-ai/mongo-predict"
    pip install "pymongopredict[sktime] @ git+https://github.com/voyage-ai/mongo-predict"   # + sktime for forecasting

## Demos

The demos use the MongoDB sample datasets. On Atlas you can load them from the cluster menu; for a local server the
loader copies the same collections from a public mirror. Run them from the repository root:

    uv sync --all-extras --group dev
    uv run python -m demos.load_sample_data --uri mongodb://localhost:27017 sample_mflix sample_airbnb sample_weatherdata
    uv run python -m demos.predictions_api.movie_ratings --uri mongodb://localhost:27017
    uv run python -m demos.predictions_api.predict_stage --uri mongodb://localhost:27017
    uv run python -m demos.embeddings_api.airbnb_room_type --uri mongodb://localhost:27017
    uv run python -m demos.embeddings_api.weather_timeseries --uri mongodb://localhost:27017

Add ``--embedding-model hashing`` to skip downloading the embedding model, ``--dimension 512`` for smaller embeddings,
or ``--streaming`` to fetch documents in batches. The results below use voyage-4-nano (2048 dimensions) on one H100.

### Predictions API: movie ratings (``demos/predictions_api/movie_ratings.py``)

The README's example: a model trained on the 17,527 movies of ``sample_mflix.movies`` released before 2010 from the
embedding of each movie (title, year, runtime, age rating, genres, cast, directors and plot, all in ``embedFields``),
and ``$predict`` rating the 5,914 released from 2010 on (the other ratings in the documents are not used).

| approach | average error (rating points) | within 1 point |
|---|---|---|
| always guess the average | 0.76 | 72% |
| ``$predict`` | **0.68** | **78%** |

### Predictions API: Airbnb listings (``demos/predictions_api/predict_stage.py``)

A room type model trained on every market except Porto (5,001 listings; accuracy 0.920 on held-out listings), then
``$predict`` on Porto's 554 listings: the most confident predictions with ``$sort`` on ``predictScore``, actual against
predicted with ``$group`` (accuracy 0.912), and the predictions written to ``sample_airbnb.listings_predictions`` with
``$merge``. With the embeddings cached (``MONGOPREDICT_PREDICT_CACHE_DIR`` set for ``$predict``), fitting the default
XGBoost model takes about 50 s and each pipeline under 1 s on 16 CPU threads.

| actual | predicted | n |
|---|---|---|
| Entire home/apt | Entire home/apt | 409 |
| Private room | Private room | 96 |
| Entire home/apt | Private room | 39 |
| Shared room | Private room | 5 |
| Private room | Entire home/apt | 5 |

### Embeddings API: Airbnb room type (``demos/embeddings_api/airbnb_room_type.py``)

``sample_airbnb.listingsAndReviews``, 5,555 listings, target ``room_type`` (63% entire home, 36% private room, 1.5%
shared room), 80/20 split stratified by the target.

| features / model | accuracy | balanced accuracy | macro F1 | features |
|---|---|---|---|---|
| most common class | 0.628 | 0.333 | | 0 |
| columns only, ridge classifier | 0.805 | 0.524 | 0.525 | 57 |
| columns only, gradient boosting | 0.870 | 0.618 | 0.635 | 57 |
| columns only, xgboost | 0.876 | 0.602 | 0.611 | 57 |
| embeddings only, ridge classifier | 0.893 | **0.628** | **0.661** | 2048 |
| columns + embeddings, ridge classifier | **0.915** | 0.609 | 0.609 | 2105 |
| columns + embeddings, xgboost | 0.912 | 0.626 | 0.643 | 2105 |

Columns: accommodates, bedrooms, beds, bathrooms, price, cleaning fee, guests included, minimum nights, number of
reviews, review score, market, property type, bed type, cancellation policy. Embedded fields: name, summary, space,
description, neighborhood overview, notes, house rules, amenities, property type, bed type, market.

### Embeddings API: weather observations (``demos/embeddings_api/weather_timeseries.py``)

``sample_weatherdata.data``, 10,000 surface observations from 5 to 13 March 1984. Regression of
``airTemperature.value`` (the 999.9 missing code filtered out), trained on the first 80% of the period (7,541
observations) and tested on the last 20% (1,885):

| features | R² | MAE (°C) | RMSE (°C) | features |
|---|---|---|---|---|
| predict the training mean | -0.001 | 9.209 | 10.708 | 0 |
| columns only, gradient boosting | 0.896 | **1.889** | 3.452 | 112 |
| columns only, xgboost | **0.901** | 1.953 | **3.371** | 112 |
| embedding of the whole document, gradient boosting | 0.733 | 3.991 | 5.527 | 2048 |
| embedding of the whole document, xgboost | 0.746 | 3.871 | 5.391 | 2048 |
| columns + embedding, gradient boosting | 0.890 | 2.020 | 3.555 | 2160 |
| columns + embedding, xgboost | 0.888 | 2.238 | 3.583 | 2160 |

The dew point column carries most of the signal, so the chosen columns do best here; the embedding of the raw
document alone still explains three quarters of the variance. Forecasting the hourly mean air temperature over all
stations (199 hours, the last 24 held out) with sktime:

| sktime forecaster | MAE (°C) | RMSE (°C) |
|---|---|---|
| last value | 7.084 | 8.210 |
| mean of the last 24 hours | 3.869 | 4.548 |
| seasonal naive (24 hours) | **2.888** | **3.657** |
| Holt-Winters additive (24 hours) | 3.068 | 3.770 |
| linear trend | 4.079 | 4.719 |

## Reports

``reports/`` holds two benchmarks, each rendered from a results file by a script in ``src/``:

| report | contents | generator |
|---|---|---|
| ``00_embeddings_api.html`` | speed of the embeddings API on the 14 sample tasks, in memory and streaming, before and after the performance work (median 2.4x faster wrapper stages, 3.4x faster fit) | ``src/speed.py``, ``src/embeddings_api_report.py`` |
| ``01_predictions_api.html`` | the ``$predict`` stage: design, overhead and agreement on the 14 sample tasks, the voyage-4-nano Porto example | ``src/predictions_api_bench.py``, ``src/predictions_api_report.py`` |

## Repository layout

| path | contents |
|---|---|
| ``pymongopredict/embeddings_api/`` | ``dataset.py`` (``MongoDataset``), ``features.py`` (``DocumentFeaturizer``), ``embed.py`` (embedding models and the cache), ``documents.py`` (paths, JSON, fetching), ``forecast.py`` (``MongoSeries``) |
| ``pymongopredict/predictions_api/`` | ``models.py`` (prediction models), ``stage.py`` (the ``$predict`` stage, ``enable()``) |
| ``tests/`` | tests of each API; ``tests/conftest.py`` holds the shared mongomock collections |
| ``demos/`` | demos of each API, run as ``python -m demos.<api>.<name>``; ``demos/load_sample_data.py`` loads the sample data |
| ``docs/`` | these pages, generated from the docstrings (``bash docs/build_docs.sh``), and the README's figures (``docs/figures/``) |
| ``reports/``, ``src/``, ``results/`` | the benchmark reports, their generators and their result files |

## Development

    uv sync --all-extras --group dev
    uv run pytest -n 4                                              # tests on mongomock, no network or GPU needed
    MONGOPREDICT_TEST_URI=mongodb://localhost:27017 uv run pytest   # also the tests against a real server
    uv run ruff check .
    bash docs/build_docs.sh                                         # rebuild these pages

- voyage-4-nano is loaded through sentence-transformers with ``trust_remote_code=True``; its code needs
  ``transformers<5``, which the package pins.
- Set ``HF_HOME`` if the default Hugging Face cache is not writable.

Examples
--------
Usage:

    from pymongopredict import PredictionModel, predictions_api

    predictions_api.enable()  # coll.aggregate() now understands $predict
    movies.create_prediction_model(PredictionModel(
        {"target": "imdb.rating", "embedFields": ["plot", "genres"]}, name="imdb_rating",
    ))
    movies.aggregate([
        {"$predict": {"model": "imdb_rating", "filter": {"year": {"$gte": 2010}}}},
        {"$sort": {"p": {"$meta": "prediction"}}},   # best predicted first
        {"$limit": 10},
        {"$project": {"title": 1, "predicted_rating": {"$meta": "prediction"}}},
    ])
"""

from . import embeddings_api, predictions_api
from .embeddings_api import *  # noqa: F403
from .predictions_api import *  # noqa: F403

__version__ = "0.5.0"
__all__ = ["embeddings_api", "predictions_api", *embeddings_api.__all__, *predictions_api.__all__]
