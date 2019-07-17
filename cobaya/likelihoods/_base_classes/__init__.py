import os
import logging
import sys
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


class _DataSetLikelihood(Likelihood):
    """A likelihood reading parameters and filenames from a .dataset plain text .ini file (as CosmoMC)"""

    data_name = ""
    supp_data_name = "planck_supp_data_and_covmats"
    supp_data_version = "v1.0"

    fast_chi_squared = _fast_chi_square()

    def initialize(self):

        if os.path.isabs(self.dataset_file):
            data_file = self.dataset_file
        else:
            # If no path specified, use the modules path
            if self.path:
                data_file_path = self.path
            elif self.path_install:
                data_file_path = self.get_path(self.path_install)
            else:
                raise LoggedError(self.log,
                                  "No path given for %s. Set the likelihood property 'path' "
                                  "or the common property '%s'.", self.dataset_file, _path_install)

            data_file = os.path.normpath(os.path.join(data_file_path, self.dataset_file))
        try:
            self.load_dataset_file(data_file, self.dataset_params)
        except IOError:
            raise LoggedError(self.log, "The data file '%s' could not be found at '%s'. "
                                        "Check your paths! %s,%s", self.dataset_file, data_file_path,
                              os.listdir(data_file_path), os.listdir(os.path.join(data_file_path, 'CamSpec2018')))

    def load_dataset_file(self, filename, dataset_params):
        ini = IniFile(filename)
        ini.params.update(dataset_params or {})
        self.init_params(ini)

    def init_params(self, ini):
        assert False, "set_file_params should be inherited"

    @classmethod
    def is_installed(cls, **kwargs):
        return os.path.exists(cls.get_path(kwargs["path"]))

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(os.path.join(path, "data", cls.data_name))

    @classmethod
    def install(cls, path=None, force=False, code=False, data=True, no_progress_bars=False):
        if not data:
            return True
        from cobaya.install import download_github_release
        log = logging.getLogger(__name__.split(".")[-1])
        log.info("Downloading %s data..." % cls.data_name)
        return download_github_release(os.path.join(path, "data"), cls.supp_data_name,
                                       cls.supp_data_version, no_progress_bars=no_progress_bars)
