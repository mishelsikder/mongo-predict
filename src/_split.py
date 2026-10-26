"""Train/test splits of the report tasks (``Task.split`` in src/tasks.py) with scikit-learn.

Two kinds of spec are used: ``{"test_size", "stratify": True, "random_state"}`` (stratified by the
target) and ``{"time_field", "test_size"}`` (the newest documents form the test part; documents
without the field are left out).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from pymongopredict.embeddings_api._time import _as_datetime


def split(ds, test_size=0.2, stratify=False, random_state=0, time_field=None):
    """``(train, test)`` MongoDataset subsets of ``ds`` for one task spec."""
    if time_field is None:
        return train_test_split(
            ds, test_size=test_size, stratify=ds.y if stratify else None, random_state=random_state
        )
    times = [_as_datetime(v) for v in ds.field_values(time_field)]
    have = [i for i, t in enumerate(times) if t is not None]
    order = np.array(sorted(have, key=lambda i: times[i]), dtype=int)  # oldest first, stable
    tr, te = train_test_split(order, test_size=test_size, shuffle=False)
    return ds.subset(np.sort(tr)), ds.subset(np.sort(te))
