import os
import logging
from getdist import IniFile

# Local
from cobaya.conventions import _path_install
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError


class _fast_chi_square(object):
    def __get__(self, instance, owner):
        # delay testing active camb until run time
        try:
            from camb.mathutils import chi_squared as fast_chi_squared
        except:
            def fast_chi_squared(covinv, x):
                return covinv.dot(x).dot(x)

        instance.fast_chi_squared = fast_chi_squared
        return fast_chi_squared


class _InstallableLikelihood(Likelihood):
    install_options = {"github_repository": "CobayaSampler/planck_supp_data_and_covmats", "github_release": "master"}

    @classmethod
    def get_install_options(cls):
        return cls.install_options

    @classmethod
    def get_path(cls, path):
        opts = cls.get_install_options()
        repo = opts.get("github_repository", None)
        if repo:
            data_path = repo.split('/')[-1]
        else:
            data_path = opts.get("data_path", cls.__name__)
        return os.path.realpath(os.path.join(path, "data", data_path))

    @classmethod
    def is_installed(cls, **kwargs):
        if kwargs["data"]:
            return os.path.exists(cls.get_path(kwargs["path"]))
        return True

    @classmethod
    def install(cls, path=None, force=False, code=False, data=True, no_progress_bars=False):
        if not data:
            return True
        log = logging.getLogger(cls.__name__)
        opts = cls.get_install_options()
        repo = opts.get("github_repository", None)
        if repo:
            from cobaya.install import download_github_release
            log.info("Downloading %s data..." % repo)
            return download_github_release(os.path.join(path, "data"), repo, opts.get("github_release", "master"),
                                           no_progress_bars=no_progress_bars)
        else:
            full_path = cls.get_path(path)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            if not data:
                return True
            filename = opts["download_url"]
            log.info("Downloading likelihood data file: %s...", filename)
            from cobaya.install import download_file
            if not download_file(filename, full_path, decompress=True, logger=log,
                                 no_progress_bars=no_progress_bars):
                return False
            log.info("Likelihood data downloaded and uncompressed correctly.")
            return True


class _DataSetLikelihood(_InstallableLikelihood):
    """A likelihood reading parameters and filenames from a .dataset plain text .ini file (as CosmoMC)"""

    default_dataset_params = {}

    fast_chi_squared = _fast_chi_square()

    def initialize(self):

        if os.path.isabs(self.dataset_file):
            data_file = self.dataset_file
            self.path = os.path.split(data_file)[0]
        else:
            # If no path specified, use the modules path
            if not self.path and self.path_install:
                self.path = self.get_path(self.path_install)
            if not self.path:
                raise LoggedError(self.log,
                                  "No path given for %s. Set the likelihood property 'path' "
                                  "or the common property '%s'.", self.dataset_file, _path_install)

            data_file = os.path.normpath(os.path.join(self.path, self.dataset_file))
        if not os.path.exists(data_file):
            raise LoggedError(self.log, "The data file '%s' could not be found at '%s'. "
                                        "Check your paths!", self.dataset_file, self.path)
        self.load_dataset_file(data_file, self.dataset_params)

    def load_dataset_file(self, filename, dataset_params):
        if '.dataset' not in filename:
            filename += '.dataset'
        ini = IniFile(filename)
        self.dataset_filename = filename
        ini.params.update(self.default_dataset_params)
        ini.params.update(dataset_params or {})
        self.init_params(ini)

    def init_params(self, ini):
        assert False, "set_file_params should be inherited"
