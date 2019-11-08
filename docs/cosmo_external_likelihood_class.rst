Creating your own cosmological likelihood class
===============================================

Creating new internal likelihoods or external likelihood classes should be straightforward.
For simple cases you can also just define a likelihood function (see :doc:`cosmo_external_likelihood`).

Likelihoods should inherit from the base :class:`.likelihood.Likelihood` class, or one of the existing extensions.
A minimal framework would look like this

.. code:: python

    from cobaya.likelihood import Likelihood
    import numpy as np
    import os

    class my_likelihood(Likelihood):

        def initialize(self):
            """
             Prepare any computation, importing any necessary code, files, etc.

             e.g. here we load some data file, with default cl_file set in .yaml below,
             or overriden when running cobaya.
            """

            self.data = np.loadtxt(self.cl_file)

        def add_theory(self):
            """
             Specifying which quantities calculated by a theory code are needed
             by calling self.theory.needs.

             e.g. here we need C_L^{tt} to lmax=2500 and the H0 value
            """
            self.theory.needs(Cl={'tt': 2500}, H0=None)

        def logp(self, **params_values):
            """
            Taking a dictionary of (sampled) nuisance parameter values params_values
            and return a log-likelihood.

            e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
            """
            H0_theory = self.theory.get_param("H0")
            cls = self.theory.get_Cl(ell_factor=True)
            my_foreground_amp = params_values['my_foreground_amp']

            chi2 = ...
            return -chi2/2


You can also implement an optional ``close`` method doing whatever needs to be done at the end of the sampling (e.g. releasing memory).

The default settings for your likelihood are specified in a ``my_likelihood.yaml`` file in the same folder as the class module, for example


.. code:: yaml

    likelihood:
      mylikes.my_likelihood:
        cl_file: /path/do/data_file
        # Aliases for automatic covariance matrix
        renames: [myOld]
        # Speed in evaluations/second (after theory inputs calculated).
        speed: 500
    params:
      my_foreground_amp:
        prior:
          dist: uniform
          min: 0
          max: 100
        ref:
          dist: norm
          loc: 153
          scale: 27
        proposal: 27
        latex: A^{f}_{\rm{mine}}


Note that if you have several nuisance parameters, fast-slow samplers will benefit from making your
likelihood faster even if it is already fast compared to the theory calculation.
If it is more than a few milliseconds consider recoding more carefully or using numba where needed.

Many real-world examples are available in cobaya.likelihoods, which you may be able to adapt as needed for more
complex cases, and a number of base class are pre-defined that you may find useful to inherit from instead of Likelihood directly.

_InstallableLikelihood
-------------------------

This supports the default auto-installation. Just add a class-level string specifying installation options, e.g.

.. code:: python

    from cobaya.likelihoods._base_classes import _InstallableLikelihood

    class my_likelihood(_InstallableLikelihood):
        install_options = {"github_repository": "MyGithub/my_repository",
                           "github_release": "master"}

        ...


You can also use install_options = {"download_url":"..url.."}

_DataSetLikelihood
-------------------

This inherits from *_InstallableLikelihood* and wraps loading settings from a .ini-format .dataset file giving setting
related to the likelihood (specified as *dataset_file* in the input .yaml).

.. code:: python

    from cobaya.likelihoods._base_classes import _DataSetLikelihood

    class my_likelihood(_DataSetLikelihood):

        def init_params(self, ini):
            """
            Load any settings from the .dataset file (ini).

            e.g. here load from "cl_file=..." specified in the dataset file
            """

            self.cl_data = np.load_txt(ini.string('cl_file'))
        ...


_cmblikes_prototype
--------------------

This the *CMBlikes* self-describing text .dataset format likelihood inherited from *_DataSetLikelihood* (as used by the
Bicep and Planck lensing likelihoods). This already implements the calculation of Gaussian and Hammimeche-Lewis
likelihoods from binned C_L data, so in simple cases you don't need to override anything, you just supply the
.yaml and .dataset file (and corresponding references data and covariance files).
Extensions and optimizations are welcome as pull requests.

.. code:: python

    from cobaya.likelihoods._base_classes import _cmblikes_prototype

    class my_likelihood(_cmblikes_prototype):
        install_options = {"github_repository": "CobayaSampler/planck_supp_data_and_covmats"}
        pass

For example *planck_2018_lensing.native* (which is installed as an internal likelihood) has this .yaml file

.. code:: yaml

    likelihood:
      planck_2018_lensing.native:
        # Path to the data: where the planck_supp_data_and_covmats has been cloned
        path: null
        dataset_file: lensing/2018/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.dataset
        # Overriding of .dataset parameters
        dataset_params:
          # field: value
        field_names: [T, E, B, P]
        # Overriding of the maximum ell computed
        l_max:
        # Aliases for automatic covariance matrix
        renames: [lensing]
        # Speed in evaluations/second
        speed: 50

    params: !defaults [../planck_2018_highl_plik/params_calib]

The description of the data files and default settings are in the `dataset file <https://github.com/CobayaSampler/planck_supp_data_and_covmats/blob/master/lensing/2018/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.dataset>`_.
The :class:`bicep_keck_2015` likelihood provides a more complicated model that adds methods to implement the foreground model.

This example also demonstrates how to share nuisance parameter settings between likelihoods: in this example all the
Planck likelihoods depend on the calibration parameter, where here the default settings for that are loaded from the
.yaml file under *planck_2018_highl_plik*.

Real-world examples
--------------------

The simplest example are the :class:`_H0_prototype` likelihoods, which are just implemented as simple Gaussians.

For an examples of more complex real-world CMB likelihoods, see :class:`bicep_keck_2015` and the lensing likelihood shown above (both
using CMBlikes format), or :class:`_planck_2018_CamSpec_python` for a full Python implementation of the
multi-frequency Planck likelihood (based from *_DataSetLikelihood*). The :class:`_planck_pliklite_prototype`
likelihood implements the plik-lite foreground-marginalized likelihood. Both the plik-like and CamSpec likelihoods
support doing general multipole and spectrum cuts on the fly by setting override dataset parameters in the input .yaml.

The provided BAO likelihoods base from :class:`_bao_prototype`, reading from simlpe text files.

The  :class:`_des_prototype` likelihood (based from *_DataSetLikelihood*) implements the DES Y1 likelihood, using the
matter power spectra to calculate shear, count and cross-correlation angular power spectra internally.

