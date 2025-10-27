CMB from Planck
==================

Family of Planck CMB likelihoods. Contains interfaces to the official 2018 ``clik`` code and some
native ones, including more recent NPIPE (PR4) results that can be run without clik.
You can find a description of the different original likelihoods in the
`Planck wiki <https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code>`_.

.. |br| raw:: html

   <br />

The Planck 2018 baseline likelihoods defined here are:

- ``planck_2018_lowl.[TT|EE]``: low-:math:`\ell` temperature-only or EE polarization-only (Cobaya-native python implementation)
- ``planck_2018_lowl.[TT|EE]_clik``: original ``clik`` versions of the above
- ``planck_2018_highl_plik.[TT|TE|EE|TTTEEE]``: ``plikHM`` high-:math:`\ell` temperature|polarization|temperature+polarization (``clik`` version)
- ``planck_2018_highl_plik.[TT|TTTEEE]_unbinned``: unbinned versions of the previous ones (``clik`` version)
- ``planck_2018_highl_plik.[TT|TTTEEE]_lite``: faster nuisance-marginalized versions of the previous (binned) ones (``clik`` version)
- ``planck_2018_highl_plik.[TT|TTTEEE]_lite_native``: Python Cobaya-native versions of the nuisance-marginalizes ones (more customizable)
- ``planck_2018_highl_CamSpec.[TT|TTTEEE]``: Cobaya-native Python versions of the alternative high-:math:`\ell` ``CamSpec`` likelihoods.
- ``planck_2018_lensing.clik``: lensing temperature+polarization-based (``clik`` version)
- ``planck_2018_lensing.native``: lensing temperature+polarization-based; Cobaya-native Python version (more customizable)
- ``planck_2018_lensing.CMBMarged``: CMB-marginalized, temperature+polarization-based lensing likelihood; Cobaya-native Python version (more customizable). Do not combine with any of the ones above!

.. note::

   **If you use any of these likelihoods, please cite and other relevant papers:**
   |br|
   Planck Collaboration, `Planck 2018 results. V. CMB power spectra and likelihoods`
   `(arXiv:1907.12875) <https://arxiv.org/abs/1907.12875>`_
   |br|
   Planck Collaboration, `Planck 2018 results. VIII. Gravitational lensing`
   `(arXiv:1807.06210) <https://arxiv.org/abs/1807.06210>`_

Other more recent Planck likelihoods are:

- ``planck_NPIPE_highl_CamSpec.[TT|TE|EE|TTEE|TTTEEE]``: latest native (bundled) python NPIPE (PR4) CamSpec high-:math:`\ell` likelihoods
- ``planck_2020_lollipop.[lowlE|lowlB|lowlEB]``: latest python NPIPE (PR4) Lollipop low-:math:`\ell` likelihoods. pip install from `GitHub <https://github.com/planck-npipe/lollipop>`__
- ``planck_2020_hillipop.[TT|TE|EE|TTTEEE]``: latest python NPIPE (PR4) Hillipop high-:math:`\ell` likelihoods. pip install from `GitHub <https://github.com/planck-npipe/hillipop>`__
- ``planckpr4lensing.[PlanckPR4Lensing|PlanckPR4LensingMarged]``: NPIPE lensing; pip install from `GitHub <https://github.com/carronj/planck_PR4_lensing>`__
- ``planck_2018_highl_CamSpec2021.[TT|TTTEEE]``: native Python versions of high-:math:`\ell` ``CamSpec`` likelihoods (from `arXiv 1910.00483 <https://arxiv.org/abs/1910.00483>`_).
- ``planck_2018_lowl.EE_sroll2``: low-:math:`\ell` EE polarization from 2019 Sroll2 analysis (native python)

Usage
-----

To use any of the Planck likelihoods, you simply need to mention them in the
``likelihood`` block, or add them using the :doc:`input generator <cosmo_basic_runs>`.

The corresponding nuisance parameters will be added automatically,
so you don't have to care about listing them in the ``params`` block.

The nuisance parameters and their default priors can be obtained as explained in :ref:`citations`.

Customization of ``clik``-based likelihoods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Likelihoods based on the official ``clik`` code (``planck_2018_lowl.[TT|EE]_clik``, ``planck_2018_highl_plik.[TT|TE|EE|TTTEEE]][_unbinned|_lite]``, ``planck_2018_highl_plik.[TT|TTTEEE]``, ``planck_2018_lensing.clik``) now use the pure-Python  `clipy implementation <https://github.com/benabed/clipy>`_.

These likelihoods can now take commands at initialization for cropping, notching or spectra from individual frequency maps. The syntax can be found in the ``README`` section of the `Github repo <https://github.com/benabed/clipy>`__.

To use any of this commands in Cobaya, simply pass a single one or a list of them, e.g.:

.. code:: yaml

   likelihood:
     planck_2018_highl_plik.TTTEEE:
       commands: ["no TT", "only EE 217x217 500 800 lax"]

Installation
------------

This likelihoods can be installed automatically as explained in :doc:`installation_cosmo`.
If you are following the instructions there (you should!), you don't need to read the rest
of this section.

Manual installation of the Planck 2018 likelihoods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you would like to install the Python-native ``clipy`` implementation by hand (e.g. with the purpose of modifying it), you can simply clone its `Github repo <https://github.com/benabed/clipy>`__ anywhere and install this local copy with ``pip install .``. Alternatively, you can automatically install the last release with ``pip install clipy-like`` (NB: ``pip install clipy`` installs a completely different package!). In both cases, make sure that you can ``import clipy`` from anywhere.

Now, download the required likelihood files from the `Planck Legacy Archive <https://pla.esac.esa.int/pla/#cosmology>`_. For instance, if you want to reproduce the baseline Planck 2018 results, download the file ``COM_Likelihood_Data-baseline_R3.00.tar.gz`` and decompress it under e.g. ``/your/path/to/planck_2018``.

Finally, download and decompress in that same folder the last release at
`this repo <https://github.com/CobayaSampler/planck_supp_data_and_covmats/releases>`_.

You can now invoke the Planck 2018 likelihoods as

.. code-block:: yaml

   likelihood:
     planck_2018_lowl.TT_clik:
       clik_file: /your/path/to/planck_2018/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik
     planck_2018_highl_plik.TTTEEE:
       clik_file: /your/path/to/planck_2018/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik


Interface for official ``clik`` code
------------------------------------

.. automodule:: cobaya.likelihoods.base_classes.planck_clik
   :noindex:

Native ``CamSpec`` version
--------------------------

.. automodule:: cobaya.likelihoods.base_classes.planck_2018_CamSpec_python
   :noindex:

Native ``lite`` version
-----------------------

.. automodule:: cobaya.likelihoods.base_classes.planck_pliklite
   :noindex:
