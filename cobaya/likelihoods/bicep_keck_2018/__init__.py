"""
.. module:: bicep_keck_2018

:Synopsis: Likelihood of BICEP/Keck, October 2021 (2018 data)
:Author: BICEP/Keck Collaboration and Antony Lewis

.. |br| raw:: html

   <br />

.. note::

   **If you use this likelihood, please cite it as:**
   |br|
   Keck Array and BICEP2 Collaborations,
   `BICEP / Keck XIII: Improved Constraints on Primordial Gravitational Waves
   using Planck, WMAP, and BICEP/Keck Observations through the 2018 Observing Season`
   `(arXiv:2110.00483v1) <https://arxiv.org/abs/2110.00483>`_

Usage
-----

To use this likelihood, ``bicep_keck_2018``, you simply need to mention it in the
``likelihood`` block, or add them using the :doc:`input generator <cosmo_basic_runs>`.

The corresponding nuisance parameters will be added automatically,
so you don't have to care about listing them in the ``params`` block.

The nuisance parameters and their default priors are reproduced below, in the *defaults*
``yaml`` file.

You shouldn't need to modify any of the options of this simple likelihood,
but if you really need to, just copy the ``likelihood`` block into your input ``yaml``
file and modify whatever options you want (you can delete the rest).

.. literalinclude:: ../cobaya/likelihoods/bicep_keck_2018/bicep_keck_2018.yaml
   :language: yaml


Installation
------------

This likelihood can be installed automatically as explained in :doc:`installation_cosmo`.
If are following the instructions there (you should!), you don't need to read the rest
of this section.

Manual installation of the BICEP/Keck likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming you are installing all your
likelihoods under ``/path/to/likelihoods``, simply do

.. code:: bash

   $ cd /path/to/likelihoods
   $ mkdir bicep_keck_2018
   $ cd bicep_keck_2018
   $ wget http://bicepkeck.org/BK18_datarelease/BK18_cosmomc.tgz
   $ tar xvf BK18_cosmomc.tgz
   $ rm BK18_cosmomc.tgz

After this, mention the path to this likelihood when you include it in an input file as

.. code-block:: yaml

   likelihood:
     bicep_keck_2018:
       path: /path/to/likelihoods/bicep_keck_2018

"""

from typing import Any

import numpy as np

from cobaya.conventions import Const
from cobaya.likelihoods.base_classes import CMBlikes

# Physical constants
Ghz_Kelvin = Const.h_J_s / Const.kB_J_K * 1e9
T_CMB_K = 2.7255  # fiducial CMB temperature


class bicep_keck_2018(CMBlikes):
    r"""
    CMB power spectrum likelihood of BICEP/Keck XIII, \cite{BK18}.
    """

    install_options = {
        "download_url": r"http://bicepkeck.org/BK18_datarelease/BK18_cosmomc.tgz"
    }

    lform_dust_decorr: str
    lform_sync_decorr: str

    def init_params(self, ini):
        super().init_params(ini)
        self.fpivot_dust = ini.float("fpivot_dust", 353.0)
        self.fpivot_sync = ini.float("fpivot_sync", 23.0)
        self.bandpasses = []
        for i, used in enumerate(self.used_map_order):
            self.bandpasses += [
                self.read_bandpass(ini.relativeFileName("bandpass[%s]" % used))
            ]

        self.fpivot_dust_decorr = [
            ini.array_float("fpivot_dust_decorr", i, default)
            for i, default in zip([1, 2], [217.0, 353.0])
        ]
        self.fpivot_sync_decorr = [
            ini.array_float("fpivot_sync_decorr", i, default)
            for i, default in zip([1, 2], [22.0, 33.0])
        ]

    def read_bandpass(self, fname):
        bandpass: Any = Bandpass()
        bandpass.R = np.loadtxt(fname)
        nu = bandpass.R[:, 0]
        bandpass.dnu = np.hstack(
            ((nu[1] - nu[0]), (nu[2:] - nu[:-2]) / 2, (nu[-1] - nu[-2]))
        )
        # Calculate thermodynamic temperature conversion between this bandpass
        # and pivot frequencies 353 GHz (used for dust) and 150 GHz (used for sync).
        th_int = np.sum(
            bandpass.dnu
            * bandpass.R[:, 1]
            * bandpass.R[:, 0] ** 4
            * np.exp(Ghz_Kelvin * bandpass.R[:, 0] / T_CMB_K)
            / (np.exp(Ghz_Kelvin * bandpass.R[:, 0] / T_CMB_K) - 1) ** 2
        )
        nu0 = self.fpivot_dust
        th0 = (
            nu0**4
            * np.exp(Ghz_Kelvin * nu0 / T_CMB_K)
            / (np.exp(Ghz_Kelvin * nu0 / T_CMB_K) - 1) ** 2
        )
        bandpass.th_dust = th_int / th0
        nu0 = self.fpivot_sync
        th0 = (
            nu0**4
            * np.exp(Ghz_Kelvin * nu0 / T_CMB_K)
            / (np.exp(Ghz_Kelvin * nu0 / T_CMB_K) - 1) ** 2
        )
        bandpass.th_sync = th_int / th0
        # Calculate bandpass center-of-mass (i.e. mean frequency).
        bandpass.nu_bar = np.dot(
            bandpass.dnu, bandpass.R[:, 0] * bandpass.R[:, 1]
        ) / np.dot(bandpass.dnu, bandpass.R[:, 1])

        return bandpass

    def dust_scaling(self, beta, Tdust, bandpass, nu0, bandcenter_err):
        """Calculates greybody scaling of dust signal defined at 353 GHz
        to specified bandpass."""
        gb_int = np.sum(
            bandpass.dnu
            * bandpass.R[:, 1]
            * bandpass.R[:, 0] ** (3 + beta)
            / (np.exp(Ghz_Kelvin * bandpass.R[:, 0] / Tdust) - 1)
        )
        # Calculate values at pivot frequency.
        gb0 = nu0 ** (3 + beta) / (np.exp(Ghz_Kelvin * nu0 / Tdust) - 1)
        #  Add correction for band center error
        if bandcenter_err != 1:
            nu_bar = Ghz_Kelvin * bandpass.nu_bar
            # Conversion factor error due to bandcenter error.
            th_err = bandcenter_err**4 * (
                np.exp(Ghz_Kelvin * bandpass % nu_bar * (bandcenter_err - 1) / T_CMB_K)
                * (np.exp(nu_bar / T_CMB_K) - 1) ** 2
                / (np.exp(nu_bar * bandcenter_err / T_CMB_K) - 1) ** 2
            )
            # Greybody scaling error due to bandcenter error.
            gb_err = (
                bandcenter_err ** (3 + beta)
                * (np.exp(nu_bar / Tdust) - 1)
                / (np.exp(nu_bar * bandcenter_err / Tdust) - 1)
            )
        else:
            th_err = 1
            gb_err = 1

        # Calculate dust scaling.
        return (gb_int / gb0) / bandpass.th_dust * (gb_err / th_err)

    def sync_scaling(self, beta, bandpass, nu0, bandcenter_err):
        """Calculates power-law scaling of synchrotron signal defined at 150 GHz
        to specified bandpass."""
        # Integrate power-law scaling and thermodynamic temperature conversion
        # across experimental bandpass.
        pl_int = np.sum(bandpass.dnu * bandpass.R[:, 1] * bandpass.R[:, 0] ** (2 + beta))
        # Calculate values at pivot frequency.
        pl0 = nu0 ** (2 + beta)
        if bandcenter_err != 1:
            nu_bar = Ghz_Kelvin * bandpass.nu_bar
            th_err = (
                bandcenter_err**4
                * np.exp(nu_bar * (bandcenter_err - 1) / T_CMB_K)
                * (np.exp(nu_bar / T_CMB_K) - 1) ** 2
                / (np.exp(nu_bar * bandcenter_err / T_CMB_K) - 1) ** 2
            )
            pl_err = bandcenter_err ** (2 + beta)
        else:
            pl_err = 1
            th_err = 1
        # Calculate sync scaling.
        return (pl_int / pl0) / bandpass.th_sync * (pl_err / th_err)

    def decorrelation(self, delta, nu0, nu1, nupivot, rat, lform):
        # Calculate factor by which foreground (dust or sync) power is decreased
        # for a cross-spectrum between two different frequencies.

        # Decorrelation scales as log^2(nu0/nu1). rat is l/l_pivot
        scl_nu = (np.log(nu0 / nu1) ** 2) / (np.log(nupivot[0] / nupivot[1]) ** 2)
        # Functional form for ell scaling is specified in .dataset file.
        if lform == "flat":
            scl_ell = 1.0
        elif lform == "lin":
            scl_ell = rat
        elif lform == "quad":
            scl_ell = rat**2
        else:
            scl_ell = 1.0

        # Even for small cval, correlation can become negative for sufficiently
        # large frequency difference or ell value (with linear or quadratic scaling).
        # Following Vansyngel et al, A&A, 603, A62 (2017), we use an exponential function
        # to remap the correlation coefficient on to the range [0,1].
        # We symmetrically extend this function to (non-physical) correlation coefficients
        # greater than 1 -- this is only used for validation tests of the likelihood
        # model.
        if delta > 1:
            return 2.0 - np.exp(np.log(2.0 - delta) * scl_nu * scl_ell)
        else:
            return np.exp(np.log(delta) * scl_nu * scl_ell)

    def add_foregrounds(self, cls, data_params):
        lpivot = 80.0
        Adust = data_params["BBdust"]
        Async = data_params["BBsync"]
        alphadust = data_params["BBalphadust"]
        betadust = data_params["BBbetadust"]
        Tdust = data_params["BBTdust"]
        alphasync = data_params["BBalphasync"]
        betasync = data_params["BBbetasync"]
        dustsync_corr = data_params["BBdustsynccorr"]
        EEtoBB_dust = data_params["EEtoBB_dust"]
        EEtoBB_sync = data_params["EEtoBB_sync"]
        delta_dust = data_params["delta_dust"]
        delta_sync = data_params["delta_sync"]
        gamma_corr = data_params["gamma_corr"]  # 13

        # Calculate dust and sync scaling for each map.
        bandcenter_err = np.empty(self.nmaps_required)
        fdust = np.empty(self.nmaps_required)
        fsync = np.empty(self.nmaps_required)
        for i, mapname in enumerate(self.used_map_order):
            # Read and assign values to band center error params
            if "95" in mapname:
                bandcenter_err[i] = gamma_corr + data_params["gamma_95"] + 1
            elif "150" in mapname:
                bandcenter_err[i] = gamma_corr + data_params["gamma_150"] + 1
            elif "220" in mapname:
                bandcenter_err[i] = gamma_corr + data_params["gamma_220"] + 1
            else:
                bandcenter_err[i] = 1
            fdust[i] = self.dust_scaling(
                betadust, Tdust, self.bandpasses[i], self.fpivot_dust, bandcenter_err[i]
            )
            fsync[i] = self.sync_scaling(
                betasync, self.bandpasses[i], self.fpivot_sync, bandcenter_err[i]
            )

        rat = np.arange(self.pcl_lmin, self.pcl_lmax + 1) / lpivot
        dustpow = Adust * rat**alphadust
        syncpow = Async * rat**alphasync
        dustsyncpow = (
            dustsync_corr * np.sqrt(Adust * Async) * rat ** ((alphadust + alphasync) / 2)
        )

        #  Only calculate foreground decorrelation if necessary.
        need_sync_decorr = np.abs(delta_sync - 1) > 1e-5
        need_dust_decorr = np.abs(delta_dust - 1) > 1e-5
        for i in range(self.nmaps_required):
            for j in range(i + 1):
                CL = cls[i, j]
                EE = CL.theory_ij[0] == 1 and CL.theory_ij[1] == 1
                BB = CL.theory_ij[0] == 2 and CL.theory_ij[1] == 2

                if EE or BB:
                    dust = fdust[i] * fdust[j]
                    sync = fsync[i] * fsync[j]
                    dustsync = fdust[i] * fsync[j] + fsync[i] * fdust[j]

                    if EE:
                        # EE spectrum: multiply foregrounds by EE/BB ratio
                        dust *= EEtoBB_dust
                        sync *= EEtoBB_sync
                        dustsync *= np.sqrt(EEtoBB_sync * EEtoBB_dust)

                    if need_dust_decorr and i != j:
                        corr_dust = self.decorrelation(
                            delta_dust,
                            self.bandpasses[i].nu_bar * bandcenter_err[i],
                            self.bandpasses[j].nu_bar * bandcenter_err[j],
                            self.fpivot_dust_decorr,
                            rat,
                            self.lform_dust_decorr,
                        )
                    else:
                        corr_dust = 1
                    if need_sync_decorr and i != j:
                        corr_sync = self.decorrelation(
                            delta_sync,
                            self.bandpasses[i].nu_bar * bandcenter_err[i],
                            self.bandpasses[j].nu_bar * bandcenter_err[j],
                            self.fpivot_sync_decorr,
                            rat,
                            self.lform_sync_decorr,
                        )
                    else:
                        corr_sync = 1
                    #  Add foreground model to theory spectrum.
                    # NOTE: Decorrelation is not implemented for the dust/sync
                    # correlated component.
                    # In BK15, we never turned on correlation and decorrelation parameters
                    # simultaneously.
                    CL.CL += (
                        dust * dustpow * corr_dust
                        + sync * syncpow * corr_sync
                        + dustsync * dustsyncpow
                    )


class Bandpass:
    pass
