Creating your own cosmological likelihood class
===============================================

Creating new internal likelihoods or external likelihood classes should be straightforward.
For simple cases you can also just define a likelihood function (see :doc:`cosmo_external_likelihood`).

Likelihoods should inherit from the base :class:`.likelihood.Likelihood` class, or one of the existing extensions.
Note that :class:`.likelihood.Likelihood` inherits directly from :class:`.theory.Theory`, so likelihood and
theory components have a common structure, with likelihoods adding specific functions to return the likelihood.

A minimal framework would look like this

.. code:: python

    from cobaya.likelihood import Likelihood
    import numpy as np
    import os

    class MyLikelihood(Likelihood):

        def initialize(self):
            """
             Prepare any computation, importing any necessary code, files, etc.

             e.g. here we load some data file, with default cl_file set in .yaml below,
             or overridden when running Cobaya.
            """

            self.data = np.loadtxt(self.cl_file)

        def get_requirements(self):
            """
             return dictionary specifying quantities calculated by a theory code are needed

             e.g. here we need C_L^{tt} to lmax=2500 and the H0 value
            """
            return {'Cl': {'tt': 2500}, 'H0': None}

        def logp(self, **params_values):
            """
            Taking a dictionary of (sampled) nuisance parameter values params_values
            and return a log-likelihood.

            e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
            """
            H0_theory = self.provider.get_param("H0")
            cls = self.provider.get_Cl(ell_factor=True)
            my_foreground_amp = params_values['my_foreground_amp']

            chi2 = ...
            return -chi2 / 2

You can also implement an optional ``close`` method doing whatever needs to be done at the end of the sampling (e.g. releasing memory).

The default settings for your likelihood are specified in a ``MyLikelihood.yaml`` file in the same folder as the class module, for example


.. code:: yaml

    cl_file: /path/do/data_file
    # Aliases for automatic covariance matrix
    aliases: [myOld]
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


When running Cobaya, you reference your likelihood in the form ``module_name.ClassName``. For example,
if your MyLikelihood class is in a module called ``mylikes`` your input .yaml would be

.. code:: yaml

    likelihood:
      mylikes.MyLikelihood:
        # .. any parameters you want to override

If your class name matches the module name, you can also just use the module name.

Note that if you have several nuisance parameters, fast-slow samplers will benefit from making your
likelihood faster even if it is already fast compared to the theory calculation.
If it is more than a few milliseconds consider recoding more carefully or using `numba <http://numba.pydata.org/>`_ where needed.

Many real-world examples are available in cobaya.likelihoods, which you may be able to adapt as needed for more
complex cases, and a number of base class are pre-defined that you may find useful to inherit from instead of Likelihood directly.

There is no fundamental difference between internal likelihood classes (in the Cobaya likelihoods package) or those
distributed externally. However, if you are distributing externally you may also wish to provide a way to
calculate the likelihood from pre-computed theory inputs as well as via Cobaya. This is easily done by extracting
the theory results in ``logp`` and them passing them and the nuisance parameters to a separate function,
e.g. `log_likelihood` where the calculation is actually done. For example, adapting the example above to:

.. code:: python

    class MyLikelihood(Likelihood):

        ...

        def logp(self, **params_values):
            H0_theory = self.provider.get_param("H0")
            cls = self.provider.get_Cl(ell_factor=True)
            return self.log_likelihood(cls, H0, **params_values)

        def log_likelihood(self, cls, H0, **data_params)
            my_foreground_amp = data_params['my_foreground_amp']
            chi2 = ...
            return -chi2 / 2


You can then create an instance of your class and call log_likelihood, entirely independently of
Cobaya. However, in this case you have to provide the full theory results to the function, rather than using the self.provider to get them
for the current parameters (self.provider is only available in Cobaya once a full model has been instantiated).

If you want to call your likelihood for specific parameters (rather than the corresponding computed theory results), you need to
call get_model() to instantiate a full model specifying which components calculate the required theory inputs. For example,

.. code:: python


   packages_path = '/path/to/your/packages'

   info = {
       'params': fiducial_params,
       'likelihood': {'my_likelihood': MyLikelihood},
       'theory': {'camb': None},
       'packages': packages_path}

   from cobaya.model import get_model
   model = get_model(info)
   model.logposterior({'H0':71.1, 'my_param': 1.40, ...})


Input parameters can be specified in the likelihood's .yaml file as shown above.
Alternatively, they can be specified as class attributes. For example, this would
be equivalent to the .yaml-based example above

.. code:: python

    class MyLikelihood(Likelihood):
        cl_file = "/path/do/data_file"
        # Aliases for automatic covariance matrix
        aliases = ["myOld"]
        # Speed in evaluations/second (after theory inputs calculated).
        speed = 500
        params = {"my_foreground_amp":
                      {"prior": {"dist": "uniform", "min": 0, "max": 0},
                       "ref" {"dist": "norm", "loc": 153, "scale": 27},
                       "proposal": 27,
                       "latex": r"A^{f}_{\rm{mine}"}}

If your likelihood has class attributes that are not possible input parameters, they should be
made private by starting the name with an underscore.

Any class can have class attributes or a .yaml file, but not both. Class
attributes or .yaml files are inherited, with re-definitions override the inherited value.

_InstallableLikelihood
-------------------------

This supports the default auto-installation. Just add a class-level string specifying installation options, e.g.

.. code:: python

    from cobaya.likelihoods._base_classes import _InstallableLikelihood

    class MyLikelihood(_InstallableLikelihood):
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

    class MyLikelihood(_DataSetLikelihood):

        def init_params(self, ini):
            """
            Load any settings from the .dataset file (ini).

            e.g. here load from "cl_file=..." specified in the dataset file
            """

            self.cl_data = np.load_txt(ini.string('cl_file'))
        ...


_CMBlikes
--------------------

This the *CMBlikes* self-describing text .dataset format likelihood inherited from *_DataSetLikelihood* (as used by the
Bicep and Planck lensing likelihoods). This already implements the calculation of Gaussian and Hammimeche-Lewis
likelihoods from binned C_L data, so in simple cases you don't need to override anything, you just supply the
.yaml and .dataset file (and corresponding references data and covariance files).
Extensions and optimizations are welcome as pull requests.

.. code:: python

    from cobaya.likelihoods._base_classes import _CMBlikes

    class MyLikelihood(_CMBlikes):
        install_options = {"github_repository": "CobayaSampler/planck_supp_data_and_covmats"}
        pass

For example *planck_2018_lensing.native* (which is installed as an internal likelihood) has this .yaml file

.. code:: yaml

    # Path to the data: where the planck_supp_data_and_covmats has been cloned
    path: null
    dataset_file: lensing/2018/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.dataset
    # Overriding of .dataset parameters
    dataset_params:

    # Overriding of the maximum ell computed
    l_max:
    # Aliases for automatic covariance matrix
    aliases: [lensing]
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

The provided BAO likelihoods base from :class:`_bao_prototype`, reading from simple text files.

The  :class:`_des_prototype` likelihood (based from *_DataSetLikelihood*) implements the DES Y1 likelihood, using the
matter power spectra to calculate shear, count and cross-correlation angular power spectra internally.

The `example external CMB likelihood <https://github.com/CobayaSampler/planck_lensing_external>`_ is a complete example
of how to make a new likelihood class in an external Python package.

Inheritance diagram for internal cosmology likelihoods
-------------------------------------------------------

.. inheritance-diagram:: likelihoods
    :parts: 1
    :private-bases:
    :top-classes: cobaya.likelihood.Likelihood

