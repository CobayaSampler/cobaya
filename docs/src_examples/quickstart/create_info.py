from collections import OrderedDict as odict
info = {
    "likelihood": {
        "gaussian": {
            "mean": [0.2, 0],
            "cov": [[0.1, 0.05],
                    [0.05,0.2]]}},
    "params": odict([
        ("mock_a", {"prior": {"min": -0.5, "max": 3},
                    "latex": r"\alpha"}),
        ("mock_b", {"prior": {"dist": "norm", "loc": 0, "scale": 1},
                    "ref": 0,
                    "proposal": 0.5,
                    "latex": r"\beta"}),
        ("derived_a", {"latex": r"\alpha^\prime"}),
        ("derived_b", {"latex": r"\beta^\prime"})]),
    "sampler": {
        "mcmc": {"burn_in": 100, "max_samples": 1000, "learn_proposal": True}}}
