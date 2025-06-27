"""
.. module:: DataSetLikelihood

:Synopsis: Base class for .dataset based likelihoods.
:Author: Antony Lewis

"""

import os

from cobaya.component import ComponentNotInstalledError
from cobaya.conventions import packages_path_input
from cobaya.log import LoggedError
from cobaya.typing import InfoDict

from .InstallableLikelihood import InstallableLikelihood


class DataSetLikelihood(InstallableLikelihood):
    """A likelihood reading parameters and file names from a .dataset plain text
    .ini file (as CosmoMC)"""

    _default_dataset_params: InfoDict = {}

    # variables defined in yaml or input dictionary
    dataset_file: str

    def initialize(self):
        if os.path.isabs(self.dataset_file):
            data_file = self.dataset_file
            self.path = os.path.dirname(data_file)
        else:
            # If no path specified and has install options (so it installed its data as an
            # external package), use the external packages path
            if not self.path and self.get_install_options() and self.packages_path:
                self.path = self.get_path(self.packages_path)
            self.path = self.path or self.get_class_path()
            if not self.path:
                raise LoggedError(
                    self.log,
                    "No path given for %s. Set the likelihood "
                    "property 'path' or the common property '%s'.",
                    self.dataset_file,
                    packages_path_input,
                )

            data_file = os.path.normpath(os.path.join(self.path, self.dataset_file))
        if not os.path.exists(data_file):
            raise ComponentNotInstalledError(
                self.log,
                "The data file '%s' could not be found at '%s'. "
                "Either you have not installed this likelihood, "
                "or have given the wrong packages installation path.",
                self.dataset_file,
                self.path,
            )
        self.load_dataset_file(data_file, getattr(self, "dataset_params", {}))

    def load_dataset_file(self, filename, dataset_params=None):
        from getdist import IniFile

        if ".dataset" not in filename:
            filename += ".dataset"
        ini = IniFile(filename)
        self.dataset_filename = filename
        ini.params.update(self._default_dataset_params)
        ini.params.update(dataset_params or {})
        self.init_params(ini)

    def init_params(self, ini):
        assert False, "init_params should be inherited"
