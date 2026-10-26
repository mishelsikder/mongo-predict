"""One prediction task per collection of the MongoDB sample datasets.

Every task uses hand-picked ``columns`` and, for the embedding configurations, the *full*
document (``embed_fields=None``, target removed) so the benchmark answers "what does embedding
the whole JSON add on top of the columns?".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    name: str
    database: str
    collection: str
    target: str
    kind: str  # "clf" | "reg"
    columns: list[str]
    filter: dict | None = None
    split: dict = field(default_factory=lambda: {"test_size": 0.2, "random_state": 0})
    note: str = ""

    @property
    def ns(self) -> str:
        return f"{self.database}.{self.collection}"


TASKS = [
    Task(
        "airbnb_room_type",
        "sample_airbnb",
        "listingsAndReviews",
        "room_type",
        "clf",
        [
            "accommodates",
            "bedrooms",
            "beds",
            "bathrooms",
            "price",
            "cleaning_fee",
            "guests_included",
            "minimum_nights",
            "number_of_reviews",
            "review_scores.review_scores_rating",
            "address.market",
            "property_type",
            "bed_type",
            "cancellation_policy",
        ],
        split={"test_size": 0.2, "stratify": True, "random_state": 0},
        note="documents carry all reviews (median 4 KB, up to 100 KB)",
    ),
    Task(
        "weather_air_temperature",
        "sample_weatherdata",
        "data",
        "airTemperature.value",
        "reg",
        [
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
        ],
        filter={"airTemperature.value": {"$lt": 900}},
        split={"time_field": "ts", "test_size": 0.2},
        note="time based split: last 20% of the period is the test set",
    ),
    Task(
        "mflix_imdb_rating",
        "sample_mflix",
        "movies",
        "imdb.rating",
        "reg",
        [
            "year",
            "runtime",
            "num_mflix_comments",
            "imdb.votes",
            "awards.wins",
            "awards.nominations",
            "rated",
            "type",
            "genres",
            "cast",
            "countries",
            "released",
        ],
        filter={"imdb.rating": {"$type": "number"}},
    ),
    Task(
        "analytics_transaction_count",
        "sample_analytics",
        "transactions",
        "transaction_count",
        "reg",
        ["account_id", "bucket_start_date", "bucket_end_date"],
        note="largest documents in the sample data (up to 100 transactions each)",
    ),
    Task(
        "geospatial_wreck_type",
        "sample_geospatial",
        "shipwrecks",
        "feature_type",
        "clf",
        [
            "latdec",
            "londec",
            "depth",
            "watlev",
            "quasou",
            "sounding_type",
            "gp_quality",
            "chart",
            "recrd",
        ],
        filter={
            "feature_type": {
                "$in": [
                    "Wrecks - Submerged, dangerous",
                    "Wrecks - Visible",
                    "Wrecks - Submerged, nondangerous",
                ]
            }
        },
        split={"test_size": 0.2, "stratify": True, "random_state": 0},
    ),
    Task(
        "supplies_purchase_method",
        "sample_supplies",
        "sales",
        "purchaseMethod",
        "clf",
        [
            "storeLocation",
            "couponUsed",
            "customer.age",
            "customer.gender",
            "customer.satisfaction",
            "items",
            "saleDate",
        ],
        split={"test_size": 0.2, "stratify": True, "random_state": 0},
    ),
    Task(
        "companies_category",
        "sample_training",
        "companies",
        "category_code",
        "clf",
        [
            "founded_year",
            "number_of_employees",
            "deadpooled_year",
            "total_money_raised",
            "products",
            "relationships",
            "offices",
            "funding_rounds",
            "milestones",
            "competitions",
        ],
        filter={"category_code": {"$ne": None}},
        split={"test_size": 0.2, "stratify": True, "random_state": 0},
    ),
    Task(
        "grades_exam_score",
        "sample_training",
        "grades",
        "scores.0.score",
        "reg",
        ["scores.1.score", "scores.2.score", "scores.3.score", "class_id", "student_id"],
        note="100k small documents; target sits inside an array",
    ),
    Task(
        "inspections_result",
        "sample_training",
        "inspections",
        "result",
        "clf",
        ["sector", "address.city", "address.zip", "certificate_number", "date"],
        filter={
            "result": {
                "$in": [
                    "No Violation Issued",
                    "Pass",
                    "Violation Issued",
                    "Fail",
                    "Out of Business",
                ]
            }
        },
        split={"test_size": 0.2, "stratify": True, "random_state": 0},
    ),
    Task(
        "routes_codeshare",
        "sample_training",
        "routes",
        "codeshare",
        "clf",
        ["airline.name", "airline.iata", "src_airport", "dst_airport", "airplane", "stops"],
        split={"test_size": 0.2, "stratify": True, "random_state": 0},
        note="binary: codeshare flag 'Y' vs empty",
    ),
    Task(
        "stories_media",
        "sample_training",
        "stories",
        "media",
        "clf",
        [
            "topic.name",
            "container.name",
            "comments",
            "diggs",
            "submit_date",
            "promote_date",
            "status",
        ],
        split={"test_size": 0.2, "stratify": True, "random_state": 0},
    ),
    Task(
        "trips_usertype",
        "sample_training",
        "trips",
        "usertype",
        "clf",
        [
            "tripduration",
            "start station id",
            "end station id",
            "bikeid",
            "birth year",
            "gender",
            "start time",
        ],
        split={"time_field": "start time", "test_size": 0.2},
        note="time based split",
    ),
    Task(
        "tweets_user_lang",
        "sample_training",
        "tweets",
        "user.lang",
        "clf",
        [
            "user.followers_count",
            "user.friends_count",
            "user.statuses_count",
            "user.favourites_count",
            "user.time_zone",
            "user.verified",
            "source",
            "truncated",
            "entities.hashtags",
            "entities.urls",
            "entities.user_mentions",
        ],
        filter={"user.lang": {"$in": ["en", "ja", "es"]}},
        split={"test_size": 0.2, "stratify": True, "random_state": 0},
    ),
    Task(
        "zips_population",
        "sample_training",
        "zips",
        "pop",
        "reg",
        ["state", "city", "loc.0", "loc.1"],
    ),
]

BY_NAME = {t.name: t for t in TASKS}
