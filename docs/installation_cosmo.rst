Installing cosmology modules
============================

.. _install_cosmo_pre:

Pre-requisites
--------------

The only requisites are:

* Python, version ≥ 2.7
* ``pip``, the Python package manager
* (Optional) an MPI implementation (see below)
* (Optional) the necessary compilers for the cosmological codes and likelihoods

Cosmology
---------

No cosmological codes or likelihoods are installed by default. They are left to the user for the sake of lightness of this code, and in case they want to use a modified version of them.

To install the usual cosmological codes, see the corresponding *Installation* sections in names_me's documentation of the interfaces for :doc:`CAMB <theory_camb>` and :doc:`CLASS <theory_class>`.

To install the Planck 2015 likelihood, check out the *Installation* section in :doc:`likelihood_planck`.
.. _directory_structure:

Preparing a tidy directory structure
------------------------------------

After it is installed, cobaya can be called from any folder, so it is not necessary (and actually not recommended) to run your chains from within cobaya's installation folder. We recommend instead installing cobaya and all its dependencies (samplers, cosmological codes and likelihoods) in a particular folder, that we will assume is ``/path/to/cosmo``.

The final structure would look like

.. code-block:: bash

   /path/to/cosmo
            ├── cobaya
            ├── getdist
            ├── likelihoods
            │   ├── planck_2015
            │   └── [...]
            ├── CAMB
            ├── CLASS
            ├── PolyChord
            ├── [...]
            └── chains  # your chains here!

            
All the installation instructions in this documentation will be given with this directory structure in mind. Of course, it is optional, and advanced users should be able to adapt it to their particular needs.
