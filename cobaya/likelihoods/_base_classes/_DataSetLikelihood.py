"""
.. module:: _DataSetLikelihood

:Synopsis: Base class for .dataset based likelihoods.
:Author: Antony Lewis

"""
# Global
import os
from getdist import IniFile

# Local
from cobaya.conventions import _path_install
from cobaya.log import LoggedError
from ._InstallableLikelihood import _InstallableLikelihood


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


class _DataSetLikelihood(_InstallableLikelihood):
    """A likelihood reading parameters and filenames from a .dataset plain text .ini file (as CosmoMC)"""

    default_dataset_params = {}

    fast_chi_squared = _fast_chi_square()

    def initialize(self):

        if os.path.isabs(self.dataset_file):
            data_file = self.dataset_file
            self.path = os.path.dirname(data_file)
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
            raise LoggedError(
                self.log, "The data file '%s' could not be found at '%s'. "
                "Either you have not installed this likelihood, "
                "or have given the wrong modules installation path.",
                self.dataset_file, self.path)
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
