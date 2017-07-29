Likelihoods -- Planck 2015
==========================

.. automodule:: cobaya.likelihoods.planck_clik_prototype.planck_clik_prototype
   :noindex:

We recommend that you create a ``likelihoods`` folder under the folder in which you have installed cobaya and the cosmological codes, e.g. ``/path/to/cosmo_codes/``, and install your external likelihoods there.

Installation
------------

Assuming you are installing all your likelihoods under ``/path/to/cosmo/likelihoods``:

.. code:: bash

   $ cd /path/to/cosmo/likelihoods
   $ mkdir planck_2015
   $ cd planck_2015       
   $ wget http://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID=1904
   $ tar xvjf data-action?COSMOLOGY.COSMOLOGY_OID=1904
   $ rm data-action?COSMOLOGY.COSMOLOGY_OID=1904
   $ cd plc-2.0
   $ ./waf configure # options

If the last step failed, try adding the option ``--install_all_deps``. It it doesn't solve it, follow the instructions in the ``readme.md`` file in the ``plc-2.0`` folder.

If you have Intel's compiler and Math Kernel Library (MKL), you may want to also add the option ``--lapack_mkl=${MKLROOT}`` in the last line to make use of it.

If ``./waf configure`` ended successfully run ``./waf install`` in the same folder. You do **not** need to run ``clik_profile.sh``, as advised.

Now, download the required likelihood files from the `Planck Legacy Archive <http://pla.esac.esa.int/pla/#cosmology>`_ (Europe) or the `NASA/IPAC Archive <http://irsa.ipac.caltech.edu/data/Planck/release_2/software/>`_ (US). You can read a description of the different products in the `Planck wiki <https://wiki.cosmos.esa.int/planckpla2015/index.php/CMB_spectrum_%26_Likelihood_Code>`_.

For instance, if you want to reproduce the baseline Planck 2015 results, download the file ``COM_Likelihood_Data-baseline_R2.00.tar.gz`` from any of the two links above, and uncompress it under the ``planck_2015`` folder that you created above. Depending on the likelihoods that you have finally downloaded, the contents of ``planck_2015`` should look like

.. code:: bash

   /path/to/cosmo/
            └── likelihoods
                └── planck_2015/
                    ├── plc-2.0/
                    │   └── # bin/ build/ ...
                    └── plc_2.0/
                        ├── hi_l/
                        ├── lensing/
                        └── low_l/

And the folders ``hi_l``, ``low_l`` and ``lensing`` should be populated by the necessary likelihood files.

To test the proper installation of the Planck likelihoods, go to section :doc:`examples_planck`.
