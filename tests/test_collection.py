import numpy as np

from cobaya.collection import SampleCollection
from cobaya.model import get_model


def test_collection_custom_weights():
    model = get_model(
        {
            "likelihood": {"gaussian": "lambda x: -0.5 * x**2"},
            "params": {"x": {"prior": {"min": -5, "max": 5}}},
        }
    )
    collection = SampleCollection(model, name="test")
    xs = np.random.default_rng(0).normal(size=5)
    for x in xs:
        collection.add([x], logpost=model.logposterior({"x": x}))
    weights = np.array([1, 2, 3, 4, 5])
    for w in [weights, weights.astype(float), list(weights)]:
        w_before = np.array(w, copy=True)
        assert np.isclose(collection.mean(weights=w)[0], np.average(xs, weights=weights))
        assert np.isclose(
            collection.cov(weights=w)[0, 0], np.cov(xs, aweights=weights, ddof=0)
        )
        # the weights passed must not be modified
        assert np.array_equal(w, w_before)
