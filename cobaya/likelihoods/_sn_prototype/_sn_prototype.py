"""
.. module:: _sn_prototype

:Synopsis: Supernovae likelihood, from CosmoMC's JLA module, for Pantheon and JLA samples.
:Author: Alex Conley, Marc Betoule, Antony Lewis (see source for more specific authorship)

This code provides the following likelihoods:

- ``sn_pantheon``, for the Pantheon SN Ia sample (including Pan-STARRS1 MDS and others)
- ``sn_jla``, for the JLA SN Ia sample, based on joint SNLS/SDSS SN Ia data
- ``sn_jla_lite``, an alternative version of ``sn_jla``, marginalized over
  nuisance parameters

.. |br| raw:: html

   <br />

.. note::

   - If you use ``sn_pantheon``, please cite:|br|
     Scolnic, D. M. et al,
     `The Complete Light-curve Sample of Spectroscopically
     Confirmed Type Ia Supernovae from Pan-STARRS1 and
     Cosmological Constraints from The Combined Pantheon Sample`
     `(arXiv:1710.00845) <https://arxiv.org/abs/1710.00845>`_
   - If you use ``sn_jla`` or ``sn_jla_lite``, please cite:|br|
     Betoule, M. et al,
     `Improved cosmological constraints from a joint analysis
     of the SDSS-II and SNLS supernova samples`
     `(arXiv:1401.4064) <https://arxiv.org/abs/1401.4064>`_


Usage
-----

To use any of these likelihoods, simply mention them in the theory block
(do not use ``sn_jla`` and its `lite` version simultaneously), or add them
using the :doc:`input generator <cosmo_basic_runs>`.

The settings for each likelihood, as well as the nuisance parameters and their default
priors (in the ``sn_jla`` case only) can be found in the ``defaults.yaml``
files in the folder for the source code of each of these likelihoods,
and are reproduced below.

You shouldn't need to modify any of the options of these likelihoods,
but if you really need to, just copy the ``likelihood`` block into your input ``yaml``
file and modify whatever options you want (you can delete the rest).

.. literalinclude:: ../cobaya/likelihoods/sn_pantheon/sn_pantheon.yaml
   :language: yaml

.. literalinclude:: ../cobaya/likelihoods/sn_jla/sn_jla.yaml
   :language: yaml

.. literalinclude:: ../cobaya/likelihoods/sn_jla_lite/sn_jla_lite.yaml
   :language: yaml


Installation
------------

This likelihood can be installed automatically as explained in :doc:`installation_cosmo`.
If are following the instructions there (you should!), you don't need to read the rest
of this section.

Manual installation of the SN Ia likelihoods data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming you are installing all your
likelihoods under ``/path/to/likelihoods``, simply do

.. code:: bash

   $ cd /path/to/likelihoods
   $ git clone https://github.com/JesusTorrado/sn_data.git

After this, mention the path to this likelihood when you include it in an input file as

.. code-block:: yaml

   likelihood:
     sn_[pantheon|jla|jla_lite]:
       path: /path/to/likelihoods/sn_data

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

# Global
from __future__ import division, print_function
import numpy as np
import io
import os
import logging
from getdist import IniFile

# Local
from cobaya.conventions import _package, _path_install
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException
from cobaya.install import download_github_release

_twopi = 2 * np.pi


class _sn_prototype(Likelihood):

    def initialize(self):
        def relative_path(tag):
            return ini.relativeFileName(tag).replace('data/', '').replace('Pantheon/', '')

        # has_absdist = F, intrinsicdisp=0, idispdataset=False
        if not self.path:
            if self.path_install:
                from importlib import import_module
                self.path = getattr(
                    import_module(
                        _package + ".likelihoods." + self.name, package=_package),
                    "get_path")(self.path_install)
            else:
                self.log.error("No path given to the %s likelihood. Set the likelihood"
                               " property 'path' or the common property '%s'.",
                               self.dataset_file, _path_install)
                raise HandledException
        self.path = os.path.normpath(self.path)
        self.dataset_file_path = os.path.normpath(os.path.join(self.path, self.dataset_file))
        self.log.info("Reading data from %s", self.dataset_file_path)
        if not os.path.exists(self.dataset_file_path):
            self.log.error("The likelihood is not installed in the given path: "
                           "cannot find the file '%s'.", self.dataset_file_path)
            raise HandledException
        ini = IniFile(self.dataset_file_path)
        ini.params.update(self.dataset_params or {})
        self.twoscriptmfit = ini.bool('twoscriptmfit')
        if self.twoscriptmfit:
            scriptmcut = ini.float('scriptmcut', 10.)
        assert not ini.float('intrinsicdisp', 0) and not ini.float('intrinsicdisp0', 0)
        if hasattr(self, "alpha_beta_names"):
            self.alpha_name = self.alpha_beta_names[0]
            self.beta_name = self.alpha_beta_names[1]
        self.pecz = ini.float('pecz', 0.001)
        cols = None
        self.has_third_var = False
        data_file = os.path.join(self.path, ini.string("data_file"))
        self.log.debug('Reading %s' % data_file)
        supernovae = {}
        self.names = []
        ix = 0
        with io.open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '#' in line:
                    cols = line[1:].split()
                    for rename, new in zip(
                            ['mb', 'color', 'x1', '3rdvar', 'd3rdvar',
                             'cov_m_s', 'cov_m_c', 'cov_s_c'],
                            ['mag', 'colour', 'stretch', 'third_var',
                             'dthird_var', 'cov_mag_stretch',
                             'cov_mag_colour', 'cov_stretch_colour']):
                        if rename in cols:
                            cols[cols.index(rename)] = new
                    self.has_third_var = 'third_var' in cols
                    zeros = np.zeros(len(lines) - 1)
                    self.third_var = zeros.copy()
                    self.dthird_var = zeros.copy()
                    self.set = zeros.copy()
                    for col in cols:
                        setattr(self, col, zeros.copy())
                elif line.strip():
                    if cols is None:
                        self.log.error('Data file must have comment header')
                        raise HandledException
                    vals = line.split()
                    for i, (col, val) in enumerate(zip(cols, vals)):
                        if col == 'name':
                            supernovae[val] = ix
                            self.names.append(val)
                        else:
                            getattr(self, col)[ix] = np.float64(val)
                    ix += 1
        self.z_var = self.dz ** 2
        self.mag_var = self.dmb ** 2
        self.stretch_var = self.dx1 ** 2
        self.colour_var = self.dcolor ** 2
        self.thirdvar_var = self.dthird_var ** 2
        self.nsn = ix
        self.log.debug('Number of SN read: %s ' % self.nsn)
        if self.twoscriptmfit and not self.has_third_var:
            self.log.error('twoscriptmfit was set but thirdvar information not present')
            raise HandledException
        if ini.bool('absdist_file'):
            self.log.error('absdist_file not supported')
            raise HandledException
        covmats = [
            'mag', 'stretch', 'colour', 'mag_stretch', 'mag_colour', 'stretch_colour']
        self.covs = {}
        for name in covmats:
            if ini.bool('has_%s_covmat' % name):
                self.log.debug('Reading covmat for: %s ' % name)
                self.covs[name] = self._read_covmat(
                    os.path.join(self.path, ini.string('%s_covmat_file' % name)))
        self.alphabeta_covmat = (len(self.covs.items()) > 1 or
                                 self.covs.get('mag', None) is None)
        self._last_alpha = np.inf
        self._last_beta = np.inf
        self.marginalize = getattr(self, "marginalize", False)
        assert self.covs
        # jla_prep
        zfacsq = 25.0 / np.log(10.0) ** 2
        self.pre_vars = self.mag_var + zfacsq * self.pecz ** 2 * (
                (1.0 + self.zcmb) / (self.zcmb * (1 + 0.5 * self.zcmb))) ** 2
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
            self.step_width_alpha = self.marginalize_params['step_width_alpha']
            self.step_width_beta = self.marginalize_params['step_width_beta']
            _marge_steps = self.marginalize_params['marge_steps']
            self.alpha_grid = np.empty((2 * _marge_steps + 1) ** 2)
            self.beta_grid = self.alpha_grid.copy()
            _int_points = 0
            for alpha_i in range(-_marge_steps, _marge_steps + 1):
                for beta_i in range(-_marge_steps, _marge_steps + 1):
                    if alpha_i ** 2 + beta_i ** 2 <= _marge_steps ** 2:
                        self.alpha_grid[_int_points] = (
                                self.marginalize_params['alpha_centre'] +
                                alpha_i * self.step_width_alpha)
                        self.beta_grid[_int_points] = (
                                self.marginalize_params['beta_centre'] +
                                beta_i * self.step_width_beta)
                        _int_points += 1
            self.log.debug('Marignalizing alpha, beta over %s points' % _int_points)
            self.marge_grid = np.empty(_int_points)
            self.int_points = _int_points
            self.alpha_grid = self.alpha_grid[:_int_points]
            self.beta_grid = self.beta_grid[:_int_points]
            self.invcovs = np.empty(_int_points, dtype=np.object)
            if self.precompute_covmats:
                for i, (alpha, beta) in enumerate(zip(self.alpha_grid, self.beta_grid)):
                    self.invcovs[i] = self.inverse_covariance_matrix(alpha, beta)
        elif not self.alphabeta_covmat:
            self.inverse_covariance_matrix()

    def add_theory(self):
        # State requisites to the theory code
        self.theory.needs(**{"angular_diameter_distance": {"z": self.zcmb}})

    def _read_covmat(self, filename):
        cov = np.loadtxt(filename)
        if np.isscalar(cov[0]) and cov[0] ** 2 + 1 == len(cov):
            cov = cov[1:]
        return cov.reshape((self.nsn, self.nsn))

    def inverse_covariance_matrix(self, alpha=0, beta=0):
        if 'mag' in self.covs:
            invcovmat = self.covs['mag'].copy()
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
            if 'stretch' in self.covs:
                invcovmat += alphasq * self.covs['stretch']
            if 'colour' in self.covs:
                invcovmat += betasq * self.covs['colour']
            if 'mag_stretch' in self.covs:
                invcovmat += 2 * alpha * self.covs['mag_stretch']
            if 'mag_colour' in self.covs:
                invcovmat -= 2 * beta * self.covs['mag_colour']
            if 'stretch_colour' in self.covs:
                invcovmat -= 2 * alphabeta * self.covs['stretch_colour']
            delta = (self.pre_vars + alphasq * self.stretch_var +
                     betasq * self.colour_var + 2.0 * alpha * self.cov_mag_stretch +
                     -2.0 * beta * self.cov_mag_colour +
                     -2.0 * alphabeta * self.cov_stretch_colour)
        else:
            delta = self.pre_vars
        np.fill_diagonal(invcovmat, invcovmat.diagonal() + delta)
        self.invcov = np.linalg.inv(invcovmat)
        return self.invcov

    def alpha_beta_logp(self, lumdists, alpha=0, beta=0, invcovmat=None):
        if self.alphabeta_covmat:
            alphasq = alpha * alpha
            betasq = beta * beta
            alphabeta = alpha * beta
            invvars = 1.0 / (self.pre_vars + alphasq * self.stretch_var +
                             betasq * self.colour_var +
                             2.0 * alpha * self.cov_mag_stretch -
                             2.0 * beta * self.cov_mag_colour -
                             2.0 * alphabeta * self.cov_stretch_colour)
            wtval = np.sum(invvars)
            estimated_scriptm = np.sum((self.mag - lumdists) * invvars) / wtval
            diffmag = (self.mag - lumdists + alpha * self.stretch -
                       beta * self.colour - estimated_scriptm)
            if invcovmat is None:
                invcovmat = self.inverse_covariance_matrix(alpha, beta)
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
            chi2 = (amarg_A + np.log(amarg_E / _twopi) +
                    np.log(tempG / _twopi) - amarg_C * amarg_C / tempG -
                    amarg_B * amarg_B * amarg_F / (amarg_E * tempG) +
                    2.0 * amarg_B * amarg_C * amarg_D / (amarg_E * tempG))
        else:
            amarg_B = np.sum(invvars)
            amarg_E = np.sum(invcovmat)
            chi2 = amarg_A + np.log(amarg_E / _twopi) - amarg_B ** 2 / amarg_E
        return - chi2 / 2

    def logp(self, **params_values):
        angular_diameter_distances = self.theory.get_angular_diameter_distance(self.zcmb)
        lumdists = (5 * np.log10((1 + self.zhel) * (1 + self.zcmb) *
                                 angular_diameter_distances))
        if self.marginalize:
            # Should parallelize this loop
            for i in range(self.int_points):
                self.marge_grid[i] = - self.alpha_beta_logp(
                    lumdists, self.alpha_grid[i],
                    self.beta_grid[i], invcovmat=self.invcovs[i])
            grid_best = np.min(self.marge_grid)
            return - grid_best + np.log(
                np.sum(np.exp(- self.marge_grid[self.marge_grid != np.inf] + grid_best)) *
                self.step_width_alpha * self.step_width_beta)
        else:
            if self.alphabeta_covmat:
                return self.alpha_beta_logp(lumdists, params_values[self.alpha_name],
                                            params_values[self.beta_name])
            else:
                return self.alpha_beta_logp(lumdists)


# Installation routines ##################################################################

# name of the data and covmats repo/folder
sn_data_name = "sn_data"
sn_data_version = "v1.3"


def get_path(path):
    return os.path.realpath(os.path.join(path, "data", sn_data_name))


def is_installed(**kwargs):
    return os.path.exists(os.path.realpath(
        os.path.join(kwargs["path"], "data", sn_data_name)))


def install(path=None, force=False, code=False, data=True, no_progress_bars=False):
    if not data:
        return True
    log = logging.getLogger(__name__.split(".")[-1])
    log.info("Downloading Supernovae data...")
    return download_github_release(os.path.join(path, "data"), sn_data_name,
                                   sn_data_version, no_progress_bars=no_progress_bars)
