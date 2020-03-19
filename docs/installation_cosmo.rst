Installing cosmological codes and data
======================================

To keep it light, maintainable and easily extensible, **cobaya** does not include code or data of any of the cosmological modules used. Instead, it provides interfaces and *automatic installers* for them.

.. _basic_requisites:

Installing a basic set of codes and likelihoods
-----------------------------------------------

To install a comprehensive set of cosmology modules (CAMB, CLASS, Planck, BICEP-Keck, BAO, SN), in a ``/path/to/modules`` folders of your choice:

.. code:: bash

   $ cobaya-install cosmo -m /path/to/modules

If this fails (see last printed message), keep on reading this section. Otherwise, you can go straight to :doc:`cosmo_basic_runs`.
   

.. _install_ext_pre:

Pre-requisites
--------------

On top of the pre-requisites of **cobaya**, you will need some others, which are indicated in the documentation of each of the modules. You may already fulfil them, so you may try to go ahead with the installation process and just take a look at the pre-requisites of the modules whose installation fails.

You will need an internet connection with a decent bandwidth (don't use your phone's): you may need to download several gigabytes!


.. _install_auto_and_directory_structure:

Using the automatic installer
-----------------------------

The automatic installation script takes one or more input files that you intend to run, makes a list of the external modules that you will need, and downloads and installs them one by one.

You need to specify a folder where the resulting files will be placed, which for the purposes of this instructions will be called ``/path/to/modules``. This does not need to be the folder in which you will run your samples.

.. warning::

   This folder will be accessed whenever you call **cobaya**, and may take several gigabytes; so, if you are in a **cluster**, make sure that said folder is placed in a *scratch* file system with rapid access from the nodes and a generous quota (this normally means that you should avoid your cluster's ``home`` folder).

When you have prepared the relevant input files, call the automatic installation script as

.. code:: bash

   $ cobaya-install input_1.yaml input_2.yaml [etc] --modules /path/to/modules

You can skip the ``--modules`` option if a ``modules`` field is already defined in **one** of the input files.

``cobaya-install`` will save the modules installation path used into a global configuration file, so that you do not need to specify it in future calls to ``cobaya-install``, ``cobaya-run``, etc. To show the current default install path, run ``cobaya-install --show-modules``.

To override the default path in a subsequent call to ``cobaya-install`` or ``cobaya-run``, the alternatives are, in descending order of precedence:

#. add an ``--modules /override/path/to/modules`` command line argument.
#. include ``modules: /override/path/to/modules`` somewhere in your :doc:`input file <input>`.
#. define an environment variable ``COBAYA_MODULES=/override/path/to/modules`` (declare it with ``export COBAYA_MODULES=[...]``).

You can run the ``cobaya-install`` script as many times as you want and it won't download or re-install already installed modules, unless the option ``--force`` (or ``-f``) is used.

Within ``/path/to/modules``, the following file structure will be created, containing only the modules that you requested:

.. code-block:: bash

   /path/to/modules
            ├── code
            │   ├── planck
            │   ├── CAMB
            │   ├── classy
            │   ├── PolyChord
            │   └── [...]
            └── data
                ├── planck_2018
                ├── bicep_keck_2015
                └── [...]

.. note::

   To run the installer from a Python script or notebook:

   .. code:: python

      from cobaya.install import install
      install(info1, info2, [etc], path='/path/to/modules')

   where ``info[X]`` are input **dictionaries**.

   If a ``path`` is not passed, it will be extracted from the given infos (it will fail if more than one have been defined).


.. _install_manual:

Installing modules manually
---------------------------

The automatic installation process above installs each module in the simplest way possible and places as much code as possible in system folders (e.g. modules that can be installed as Python packages).

If you want to modify one of the modules (e.g. one of the theory codes) you will probably prefer to install them manually. Each module's documentation has a section on manual installation, and on how to specify your installation folder at run time. Check the relevant section of the documentation of each module.

When an installation path for a particular module is given in its corresponding input block, it takes precedence over automatic installation folder described above, so that if you already installed a version automatically, it will be ignored in favour of the manually specified one.
