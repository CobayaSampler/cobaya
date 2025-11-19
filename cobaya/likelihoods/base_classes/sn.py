"""
.. module:: sn

:Synopsis: Supernovae likelihoods.
:Author: Alex Conley, Marc Betoule, Antony Lewis, Pablo Lemos
         (see source for more specific authorship)

This code provides various supernova likelihoods, with and without calibration.

Usage
-----

To use any of these likelihoods, simply mention them in the likelihoods block
(e.g. ``sn.pantheonplus``), or add them
using the :doc:`input generator <cosmo_basic_runs>`.

Installation
------------

This likelihood can be installed automatically as explained in :doc:`installation_cosmo`.
"""

# Supernovae likelihood, from CosmoMC's JLA module. For Pantheon and JLA Supernovae,
#  History:
#  Written by Alex Conley, Dec 2006
#   aconley, Jan 2007: The OpenMP stuff was causing massive slowdowns on
#      some processors (ones with hyperthreading), so it was removed
#   aconley, Jul 2009: Added absolute distance support
#   aconley, May 2010: Added twoscriptm support
#   aconley, Apr 2011: Fix some non standard F90 usage.  Thanks to
#                       Zhiqi Huang for catching this.
#   aconley, April 2011: zhel, zcmb read in wrong order.  Thanks to
#                       Xiao Dong-Li and Shuang Wang for catching this
#   mbetoule, Dec 2013: adaptation to the JLA sample
#   AL, Mar 2014: updates for latest CosmoMC structure
#   AL, June 2014: updated JLA_marginalize=T handling so it should work
#   AL, March 2018: this python version
#   PL, May 2021: Included option to use absolute magnitude from local SNe measurements.

import os

import numpy as np

from cobaya.likelihoods.base_classes.DataSetLikelihood import DataSetLikelihood
from cobaya.log import LoggedError

_twopi = 2 * np.pi


class SN(DataSetLikelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "SN"

    install_options = {
        "github_repository": "CobayaSampler/sn_data",
        "github_release": "v1.8",
    }

    use_abs_mag: bool = False
    mag: np.ndarray
    zcmb: np.ndarray

    def init_params(self, ini):
        self.twoscriptmfit = ini.bool("twoscriptmfit")
        if self.twoscriptmfit:
            scriptmcut = ini.float("scriptmcut", 10.0)

        assert not ini.float("intrinsicdisp", 0) and not ini.float("intrinsicdisp0", 0)
        if getattr(self, "alpha_beta_names", None) is not None:
            self.alpha_name = self.alpha_beta_names[0]
            self.beta_name = self.alpha_beta_names[1]
        self.pecz = ini.float("pecz", 0.001)
        cols = None
        self.has_third_var = False
        data_file = os.path.normpath(os.path.join(self.path, ini.string("data_file")))
        self.log.debug("Reading %s" % data_file)
        supernovae = {}
        self.names = []
        ix = 0
        with open(data_file) as f:
            lines = f.readlines()
            for line in lines:
                if "#" in line:
                    cols = line[1:].split()
                    for rename, new in zip(
                        [
                            "mb",
                            "color",
                            "x1",
                            "3rdvar",
                            "d3rdvar",
                            "cov_m_s",
                            "cov_m_c",
                            "cov_s_c",
                        ],
                        [
                            "mag",
                            "colour",
                            "stretch",
                            "third_var",
                            "dthird_var",
                            "cov_mag_stretch",
                            "cov_mag_colour",
                            "cov_stretch_colour",
                        ],
                    ):
                        if rename in cols:
                            cols[cols.index(rename)] = new
                    self.has_third_var = "third_var" in cols
                    has_x0_cov = "cov_s_x0" in cols
                    zeros = np.zeros(len(lines) - 1)
                    self.third_var = zeros.copy()
                    self.dthird_var = zeros.copy()
                    self.set = zeros.copy()
                    for col in cols:
                        setattr(self, col, zeros.copy())
                elif line.strip():
                    if cols is None:
                        raise LoggedError(self.log, "Data file must have comment header")
                    vals = line.split()
                    for i, (col, val) in enumerate(zip(cols, vals)):
                        if col == "name":
                            supernovae[val] = ix
                            self.names.append(val)
                        else:
                            getattr(self, col)[ix] = np.float64(val)
                    ix += 1

        if has_x0_cov:
            sf = -2.5 / (self.x0 * np.log(10))
            self.cov_mag_stretch = self.cov_s_x0 * sf
            self.cov_mag_colour = self.cov_c_x0 * sf

        self.z_var = self.dz**2
        self.mag_var = self.dmb**2
        self.stretch_var = self.dx1**2
        self.colour_var = self.dcolor**2
        self.thirdvar_var = self.dthird_var**2
        self.nsn = ix
        self.log.debug("Number of SN read: %s " % self.nsn)
        if self.twoscriptmfit and not self.has_third_var:
            raise LoggedError(
                self.log, "twoscriptmfit was set but thirdvar information not present"
            )
        if ini.bool("absdist_file"):
            raise LoggedError(self.log, "absdist_file not supported")
        covmats = [
            "mag",
            "stretch",
            "colour",
            "mag_stretch",
            "mag_colour",
            "stretch_colour",
        ]
        self.covs = {}
        for name in covmats:
            if ini.bool("has_%s_covmat" % name):
                self.log.debug("Reading covmat for: %s " % name)
                self.covs[name] = self._read_covmat(
                    os.path.join(self.path, ini.string("%s_covmat_file" % name))
                )
        self.alphabeta_covmat = (
            len(self.covs.items()) > 1 or self.covs.get("mag", None) is None
        )
        self._last_alpha = np.inf
        self._last_beta = np.inf
        self.marginalize = getattr(self, "marginalize", False)
        assert self.covs
        # jla_prep
        zfacsq = 25.0 / np.log(10.0) ** 2
        self.pre_vars = (
            self.mag_var
            + zfacsq
            * self.pecz**2
            * ((1.0 + self.zcmb) / (self.zcmb * (1 + 0.5 * self.zcmb))) ** 2
        )
        if self.twoscriptmfit:
            A1 = np.zeros(self.nsn)
            A2 = np.zeros(self.nsn)
            A1[self.third_var <= scriptmcut] = 1
            A2[self.third_var > scriptmcut] = 1
            has_A1 = np.any(A1)
            has_A2 = np.any(A2)
            if not has_A1:
                # swap
                A1 = A2
                A2 = np.zeros(self.nsn)
                has_A2 = False
            if not has_A2:
                self.twoscriptmfit = False
            self.A1 = A1
            self.A2 = A2
        if self.marginalize:
            self.step_width_alpha = self.marginalize_params["step_width_alpha"]
            self.step_width_beta = self.marginalize_params["step_width_beta"]
            _marge_steps = self.marginalize_params["marge_steps"]
            self.alpha_grid = np.empty((2 * _marge_steps + 1) ** 2)
            self.beta_grid = self.alpha_grid.copy()
            _int_points = 0
            for alpha_i in range(-_marge_steps, _marge_steps + 1):
                for beta_i in range(-_marge_steps, _marge_steps + 1):
                    if alpha_i**2 + beta_i**2 <= _marge_steps**2:
                        self.alpha_grid[_int_points] = (
                            self.marginalize_params["alpha_centre"]
                            + alpha_i * self.step_width_alpha
                        )
                        self.beta_grid[_int_points] = (
                            self.marginalize_params["beta_centre"]
                            + beta_i * self.step_width_beta
                        )
                        _int_points += 1
            self.log.debug("Marignalizing alpha, beta over %s points" % _int_points)
            self.marge_grid = np.empty(_int_points)
            self.int_points = _int_points
            self.alpha_grid = self.alpha_grid[:_int_points]
            self.beta_grid = self.beta_grid[:_int_points]
            self.invcovs = np.empty(_int_points, dtype=object)
            if self.precompute_covmats:
                for i, (alpha, beta) in enumerate(zip(self.alpha_grid, self.beta_grid)):
                    self.invcovs[i] = self.inverse_covariance_matrix(alpha, beta)
        elif not self.alphabeta_covmat:
            self.inverse_covariance_matrix()

    def get_requirements(self):
        # State requisites to the theory code
        reqs = {"angular_diameter_distance": {"z": self.zcmb}}
        if self.use_abs_mag:
            reqs["Mb"] = None
        return reqs

    def _read_inv_covmat(self, filename):
        d = np.load(filename)
        n = d[d.files[0]][0]
        inv_cov = np.zeros((n, n))
        inv_cov[np.triu_indices(n)] = d[d.files[1]]

        # Reflect to lower triangular part to make it symmetric
        i_lower = np.tril_indices(n, -1)
        inv_cov[i_lower] = inv_cov.T[i_lower]

        return inv_cov

    def _read_covmat(self, filename):
        cov = np.loadtxt(filename)
        if np.isscalar(cov[0]) and cov[0] ** 2 + 1 == len(cov):
            cov = cov[1:]
        return cov.reshape((self.nsn, self.nsn))

    def inverse_covariance_matrix(self, alpha=0, beta=0):
        if "mag" in self.covs:
            invcovmat = self.covs["mag"].copy()
        else:
            invcovmat = 0
        if self.alphabeta_covmat:
            if np.isclose(alpha, self._last_alpha) and np.isclose(beta, self._last_beta):
                return self.invcov
            self._last_alpha = alpha
            self._last_beta = beta
            alphasq = alpha * alpha
            betasq = beta * beta
            alphabeta = alpha * beta
            if "stretch" in self.covs:
                invcovmat += alphasq * self.covs["stretch"]
            if "colour" in self.covs:
                invcovmat += betasq * self.covs["colour"]
            if "mag_stretch" in self.covs:
                invcovmat += 2 * alpha * self.covs["mag_stretch"]
            if "mag_colour" in self.covs:
                invcovmat -= 2 * beta * self.covs["mag_colour"]
            if "stretch_colour" in self.covs:
                invcovmat -= 2 * alphabeta * self.covs["stretch_colour"]
            delta = (
                self.pre_vars
                + alphasq * self.stretch_var
                + betasq * self.colour_var
                + 2.0 * alpha * self.cov_mag_stretch
                + -2.0 * beta * self.cov_mag_colour
                + -2.0 * alphabeta * self.cov_stretch_colour
            )
        else:
            delta = self.pre_vars
        np.fill_diagonal(invcovmat, invcovmat.diagonal() + delta)
        self.invcov = np.linalg.inv(invcovmat)
        return self.invcov

    def alpha_beta_logp(self, lumdists, alpha=0, beta=0, Mb=0, invcovmat=None):
        if self.alphabeta_covmat:
            if self.use_abs_mag:
                self.log.warning(
                    "You seem to be using JLA with the absolute magnitude "
                    "module. JLA uses a different calibration, the Mb "
                    "module only works with Pantheon SNe!"
                )
                estimated_scriptm = Mb + 25
            else:
                alphasq = alpha * alpha
                betasq = beta * beta
                alphabeta = alpha * beta
                invvars = 1.0 / (
                    self.pre_vars
                    + alphasq * self.stretch_var
                    + betasq * self.colour_var
                    + 2.0 * alpha * self.cov_mag_stretch
                    - 2.0 * beta * self.cov_mag_colour
                    - 2.0 * alphabeta * self.cov_stretch_colour
                )
                wtval = np.sum(invvars)
                estimated_scriptm = np.sum((self.mag - lumdists) * invvars) / wtval
            diffmag = (
                self.mag
                - lumdists
                + alpha * self.stretch
                - beta * self.colour
                - estimated_scriptm
            )
            if invcovmat is None:
                invcovmat = self.inverse_covariance_matrix(alpha, beta)
        else:
            if self.use_abs_mag:
                estimated_scriptm = Mb + 25
            else:
                invvars = 1.0 / self.pre_vars
                wtval = np.sum(invvars)
                estimated_scriptm = np.sum((self.mag - lumdists) * invvars) / wtval
            diffmag = self.mag - lumdists - estimated_scriptm
            invcovmat = self.invcov

        invvars = invcovmat.dot(diffmag)
        amarg_A = invvars.dot(diffmag)

        if self.twoscriptmfit:
            # could simplify this..
            amarg_B = invvars.dot(self.A1)
            amarg_C = invvars.dot(self.A2)
            invvars = invcovmat.dot(self.A1)
            amarg_D = invvars.dot(self.A2)
            amarg_E = invvars.dot(self.A1)
            invvars = invcovmat.dot(self.A2)
            amarg_F = invvars.dot(self.A2)
            tempG = amarg_F - amarg_D * amarg_D / amarg_E
            assert tempG >= 0
            if self.use_abs_mag:
                chi2 = amarg_A + np.log(amarg_E / _twopi) + np.log(tempG / _twopi)
            else:
                chi2 = (
                    amarg_A
                    + np.log(amarg_E / _twopi)
                    + np.log(tempG / _twopi)
                    - amarg_C * amarg_C / tempG
                    - amarg_B * amarg_B * amarg_F / (amarg_E * tempG)
                    + 2.0 * amarg_B * amarg_C * amarg_D / (amarg_E * tempG)
                )
        else:
            amarg_B = np.sum(invvars)
            amarg_E = np.sum(invcovmat)
            if self.use_abs_mag:
                chi2 = amarg_A + np.log(amarg_E / _twopi)
            else:
                chi2 = amarg_A + np.log(amarg_E / _twopi) - amarg_B**2 / amarg_E
        return -chi2 / 2

    def logp(self, **params_values):
        angular_diameter_distances = self.provider.get_angular_diameter_distance(
            self.zcmb
        )
        lumdists = 5 * np.log10(
            (1 + self.zhel) * (1 + self.zcmb) * angular_diameter_distances
        )

        if self.use_abs_mag:
            Mb = params_values.get("Mb", None)
        else:
            Mb = 0
        if self.marginalize:
            # Should parallelize this loop
            for i in range(self.int_points):
                self.marge_grid[i] = -self.alpha_beta_logp(
                    lumdists,
                    self.alpha_grid[i],
                    self.beta_grid[i],
                    Mb,
                    invcovmat=self.invcovs[i],
                )
            grid_best = np.min(self.marge_grid)
            return -grid_best + np.log(
                np.sum(np.exp(-self.marge_grid[self.marge_grid != np.inf] + grid_best))
                * self.step_width_alpha
                * self.step_width_beta
            )
        else:
            if self.alphabeta_covmat:
                return self.alpha_beta_logp(
                    lumdists,
                    params_values[self.alpha_name],
                    params_values[self.beta_name],
                    Mb,
                )
            else:
                return self.alpha_beta_logp(lumdists, Mb=Mb)

    @classmethod
    def get_file_base_name(cls) -> str:
        return cls.__dict__.get("file_base_name") or cls.__name__.lower()
