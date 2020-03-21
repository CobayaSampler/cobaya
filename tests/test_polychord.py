from flaky import flaky

from .common_sampler import body_of_test, body_of_test_speeds


### @pytest.mark.mpi


@flaky(max_runs=3, min_passes=1)
def test_polychord(packages_path, tmpdir):
    dimension = 3
    n_modes = 1
    info_sampler = {"polychord": {"nlive": 25 * dimension * n_modes}}
    body_of_test(dimension=dimension, n_modes=n_modes,
                 info_sampler=info_sampler, tmpdir=str(tmpdir), packages_path=packages_path)


@flaky(max_runs=5, min_passes=1)
    dimension = 2
def test_polychord_multimodal(packages_path, tmpdir):
    n_modes = 2
    info_sampler = {"polychord": {"nlive": 40 * dimension * n_modes}}
    body_of_test(dimension=dimension, n_modes=n_modes,
                 info_sampler=info_sampler, tmpdir=str(tmpdir), packages_path=packages_path)


@flaky(max_runs=3, min_passes=1)
def test_polychord_speeds(packages_path):
    info_polychord = {"polychord": {"oversample_power": 1}}
    body_of_test_speeds(info_polychord, packages_path=packages_path)


@flaky(max_runs=3, min_passes=1)
def test_polychord_speeds_manual(packages_path):
    info_polychord = {"polychord": {"oversample_power": 1}}
    body_of_test_speeds(info_polychord, manual_blocking=True, packages_path=packages_path)
