External likelihoods
======================

This page lists external Python modules that can be used with Cobaya. For how to write your own package see :doc:`cosmo_external_likelihood_class`.

After installing an external package, when running Cobaya reference your likelihood in the form ``[package or module name].ClassName``. For example, if your ExternalLike class is in a module called ``newlike`` your input .yaml would be

.. code:: yaml

    likelihood:
      newlike.ExternalLike:
         python_path: /path/to/newlike
         # .. any parameters you want to override

The python_path is not needed if the package has been pip installed.
External packages can also automatically be pip installed from an input yaml using ``cobaya-install``, e.g.

.. code:: yaml

    likelihood:
      planckpr4lensing:
         package_install:
             github_repository: carronj/planck_PR4_lensing
             min_version: 1.0.2

List of external packages
==========================

 * `Planck NPIPE lensing <https://github.com/carronj/planck_PR4_lensing>`_
 * `Planck NPIPE hillipop and lollipop <https://github.com/planck-npipe>`_
 * `MFLike-plik <https://github.com/simonsobs/LAT_MFLike/tree/mflike-plik>`_
 * `ACTPol DR4 <https://github.com/ACTCollaboration/pyactlike>`_
 * `ACT DR6 Lensing <https://github.com/ACTCollaboration/act_dr6_lenslike>`_
 * `SPT-SZ, SPTPol & SPT-3G <https://github.com/xgarrido/spt_likelihoods>`_
 * `cobaya-mock-cmb <https://github.com/misharash/cobaya_mock_cmb>`_
 * `pyWMAP <https://github.com/HTJense/pyWMAP>`_
 * `Mock CMB-HD <https://github.com/CMB-HD/hdlike>`_
 * `TDCOSMO strong lensing time delays <https://github.com/nataliehogg/tdcosmo_ext>`_
 * `Example - simple demo <https://github.com/CobayaSampler/example_external_likelihood>`_
 * `Example - Planck lensing <https://github.com/CobayaSampler/planck_lensing_external>`_

If you have a new likelihood and would like to add it to this list, please edit likelihood_external.rst and make a `pull request <https://github.com/CobayaSampler/cobaya/pulls>`_.
