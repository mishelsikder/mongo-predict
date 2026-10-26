"""Property-based tests (Hypothesis), kept small so the suite stays fast."""

import json
import pickle

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pymongopredict import DocumentFeaturizer
from pymongopredict.embeddings_api.documents import _assemble_json, _select_fields, _without_paths

KEYS = [
    "a",
    "b",
    "0",
    "1",
]  # numeric dict keys next to real arrays: the case hand-written tests missed
FAST = settings(max_examples=150, deadline=None)


def docs_with_unique_leaves():
    counter = iter(range(10**9))
    leaf = st.builds(lambda: f"v{next(counter)}")
    tree = st.recursive(
        leaf,
        lambda inner: (
            st.lists(inner, max_size=3) | st.dictionaries(st.sampled_from(KEYS), inner, max_size=3)
        ),
        max_leaves=12,
    )
    return st.dictionaries(st.sampled_from(KEYS), tree, min_size=1, max_size=4)


def paths_of(node, prefix=""):
    """Every dotted path in a document, with numeric parts for list positions."""
    out = []
    items = (
        node.items()
        if isinstance(node, dict)
        else enumerate(node)
        if isinstance(node, list)
        else []
    )
    for k, v in items:
        p = f"{prefix}{k}"
        out.append(p)
        out += paths_of(v, p + ".")
    return out


def reached(node, parts):
    """Leaves an excluded path reaches (MongoDB semantics: a non-index part maps over arrays)."""
    if not parts:
        return set(leaves(node))
    head, rest = parts[0], parts[1:]
    if isinstance(node, dict):
        return reached(node[head], rest) if head in node else set()
    if isinstance(node, list):
        if head.isdigit():
            return reached(node[int(head)], rest) if int(head) < len(node) else set()
        return set().union(*[reached(el, parts) for el in node]) if node else set()
    return set()


def leaves(node):
    if isinstance(node, dict):
        return [x for v in node.values() for x in leaves(v)]
    if isinstance(node, list):
        return [x for v in node for x in leaves(v)]
    return [node]


@FAST
@given(docs_with_unique_leaves(), st.data())
def test_excluded_leaves_never_survive_and_others_always_do(doc, data):
    paths = paths_of(doc)
    excluded = data.draw(st.lists(st.sampled_from(paths), max_size=3)) if paths else []
    gone = set().union(*[reached(doc, p.split(".")) for p in excluded]) if excluded else set()
    kept = set(leaves(_without_paths(doc, excluded)))
    assert not kept & gone and kept == set(leaves(doc)) - gone
    top = data.draw(st.lists(st.sampled_from(list(doc)), min_size=1, max_size=2, unique=True))
    shown = set(leaves(_select_fields(doc, top, excluded)))
    assert not shown & gone and shown == set(leaves({k: doc[k] for k in top})) - gone


@FAST
@given(
    st.lists(st.text(max_size=8), max_size=6),
    st.integers(0, 60),
    st.text(max_size=5),
)
def test_json_budget(values, budget, prefix):
    pieces = [json.dumps(f"k{i}") + ":" + json.dumps(v) for i, v in enumerate(values)]
    full = prefix + "{" + ",".join(pieces) + "}"
    out = _assemble_json(pieces, max(budget - len(prefix), 0), prefix=prefix)
    assert out.startswith(prefix)
    if budget >= len(prefix) + 2:
        assert len(out) <= budget
    if len(full) <= budget:
        assert out == full and json.loads(out[len(prefix) :]) is not None


@pytest.mark.filterwarnings("ignore:columns .* have no value")  # (generated documents)
@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.fixed_dictionaries(
            {},
            optional={
                "a": st.integers(-5, 5)
                | st.floats(allow_nan=True, allow_infinity=True)
                | st.none(),
                "c": st.fixed_dictionaries({}, optional={"d": st.sampled_from(["x", "y", "z"])}),
                "e": st.lists(st.integers(0, 9), max_size=3),
                "t": st.text(max_size=10),
            },
        ),
        min_size=2,
        max_size=8,
    )
)
def test_featurizer_is_batch_invariant_finite_and_picklable(docs):
    f = DocumentFeaturizer(
        columns=["a", "c.d", "e.0"],
        embedding_model="hashing",
        embedding_kwargs={"n_features": 8},
        cache_dir=None,
        min_frequency=1,
    )
    X = f.fit_transform(docs)
    assert X.shape == (len(docs), len(f.get_feature_names_out())) and np.isfinite(X).all()
    assert np.allclose(X, f.transform(docs))
    assert np.allclose(X, np.vstack([f.transform([d]) for d in docs]))
    assert np.allclose(X, pickle.loads(pickle.dumps(f)).transform(docs))
