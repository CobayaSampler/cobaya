Installing cosmological codes and data
======================================

To keep it light, maintainable and easily extensible, **cobaya** does not include code or data of many of the cosmological components used; instead, it provides interfaces and *automatic installers* for the *external packages* they require: the original code itself, a cosmological dataset, etc.

.. _basic_requisites:

Installing a basic set of codes and likelihoods
-----------------------------------------------

To install a comprehensive set of cosmology requisites (CAMB, CLASS, Planck, BICEP-Keck, BAO, SN), in a ``/path/to/packages`` folders of your choice:

.. code:: bash

   $ cobaya-install cosmo -m /path/to/packages

If this fails (see last printed message), keep on reading this section. Otherwise, you can go straight to :doc:`cosmo_basic_runs`.


.. _install_ext_pre:

Pre-requisites
--------------

On top of the pre-requisites of **cobaya**, you will need some others, which are indicated in the documentation of each of the components. You may already fulfil them, so you may try to go ahead with the installation process and just take a look at the pre-requisites of the components whose installation fails.

You will need an internet connection with a decent bandwidth (don't use your phone's): you may need to download several gigabytes!


.. _install_auto_and_directory_structure:

Using the automatic installer
-----------------------------

The automatic installation script takes one or more input files that you intend to run, makes a list of the external packages that you will need, and downloads and installs them one by one.

You need to specify a folder where the resulting files will be placed, which for the purposes of these instructions will be called ``/path/to/packages``. This does not need to be the folder in which you will run your samples.

.. warning::

   This folder will be accessed whenever you call **cobaya**, and may take several gigabytes; so, if you are in a **cluster**, make sure that said folder is placed in a *scratch* file system with rapid access from the nodes and a generous quota (this normally means that you should avoid your cluster's ``home`` folder).

When you have prepared the relevant input files, call the automatic installation script as

.. code:: bash

   $ cobaya-install input_1.yaml input_2.yaml [etc] --packages-path /path/to/packages

You can skip the ``--packages-path`` option if a ``packages_path`` field is already defined in **one** of the input files.

.. note::

   If you would like to skip the installation of the dependencies of some components, you can use the ``--skip "word1 word2 [...]"`` argument, where ``word[X]`` are sub-strings of the names of the corresponding components (case-insensitive), e.g. ``camb`` or ``planck``.

   If you would like to automatically skip installing external packages that are avaliable globally (e.g. if you can do ``import classy`` from anywhere) add ``--skip-global`` to the command above.

``cobaya-install`` will save the packages installation path used into a global configuration file, so that you do not need to specify it in future calls to ``cobaya-install``, ``cobaya-run``, etc. To show the current default install path, run ``cobaya-install --show-packages-path``.

To override the default path in a subsequent call to ``cobaya-install`` or ``cobaya-run``, the alternatives are, in descending order of precedence:

#. add an ``--packages-path /override/path/to/packages`` command line argument.
#. include ``packages_path: /override/path/to/packages`` somewhere in your :doc:`input file <input>`.
#. define an environment variable ``COBAYA_PACKAGES_PATH=/override/path/to/packages`` (declare it with ``export COBAYA_PACKAGES_PATH=[...]``).

You can run the ``cobaya-install`` script as many times as you want and it won't download or re-install already installed packages, unless the option ``--force`` (or ``-f``) is used.

Within ``/path/to/packages``, the following file structure will be created, containing only the packages that you requested:

.. code-block:: bash

   /path/to/packages
            ├── code
            │   ├── planck
            │   ├── CAMB
            │   ├── classy
            │   ├── PolyChordLite
            │   └── [...]
            └── data
                ├── planck_2018
                ├── bicep_keck_2015
                └── [...]

.. note::

   To run the installer from a Python script or notebook:

   .. code:: python

      from cobaya.install import install
      install(info1, info2, [etc], path='/path/to/packages')

   where ``info[X]`` are input **dictionaries**.

   If a ``path`` is not passed, it will be extracted from the given infos (it will fail if more than one have been defined).


.. _install_manual:

Installing requisites manually
------------------------------

The automatic installation process above installs all the requisites for the components used in the simplest way possible, preferring the system folders when possible (e.g. code that can be installed as a Python package).

If you want to modify one of the external packages (e.g. one of the theory codes) you will probably prefer to install them manually. Each component's documentation has a section on manual installation of its requisites, and on how to specify your installation folder at run time. Check the relevant section of the documentation of each component.

When an installation path for a particular component is given in its corresponding input block, it takes precedence over automatic installation folder described above, so that if you already installed a version automatically, it will be ignored in favour of the manually specified one.

Updating and installing specific components
--------------------------------------------

Individual likelihood or theory components can be installed using

.. code:: bash

   $ cobaya-install component_name --packages-path /path/to/packages

This will also work with your own or third-party :ref:`likelihood classes <likelihood_classes>`.
To force reinstallation of a package that is already installed, you can use the ``-f`` option, e.g. to
update an auto-installed *camb* use

.. code:: bash

   $ cobaya-install -f --packages-path /path/to/packages camb
