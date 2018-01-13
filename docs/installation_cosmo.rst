Installing cosmological codes and data
======================================

To keep it light and maintainable, and to allow for maximum customization, **cobaya** only includes some of the samplers it uses, and none the cosmological likelihoods, data or theory codes. Nonetheless, they can be downloaded and installed automatically when needed.

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

When you have prepared the interesting input files and *created* the desired folder, call the automatic installation script as

.. code:: bash

   $ cobaya-install input_1.yaml input_2.yaml [etc] --path /path/to/modules


You can run the scripts as many times as you want and it won't download or re-install already installed modules, unless the option ``--force`` (or ``-f``) is used.

Within ``/path/to/modules``, the following file structure will be created, containing only the modules that you requested:

.. code-block:: bash

   /path/to/modules
            ├── code
            │   ├── planck_2015
            │   ├── CLASS
            │   └── [...]
            └── data
                ├── planck_2015
                ├── bicep_keck_2015
                └── [...]

.. note::

   Not all automatically installed modules will be placed there; e.g. those that can be installed as a Python package (CAMB, for instance) won't leave any trace in that folder. For this reason, if you plan to modify one of the modules, it is recommended that you :ref:`install it manually <install_manual>`.


Take note of that folder in your case, here ``/path/to/modules``, and include it under the field ``path_to_modules`` somewhere in your input file (see :doc:`input` for a detail description of input files):

.. code:: yaml

   path_to_modules: /path/to/modules

or specify it using the flat ``--path /path/to/modules`` when invoking from the shell.

.. _install_manual:

Installing modules manually
---------------------------

The automatic installation process above installs each module in the simplest way possible and places as much code as possible in system folders (e.g. modules that can be installed as Python packages).

If you want to modify one of the modules (e.g. one of the theory codes) you will probably prefer to install them manually. Each module's documentation has a section on manual installation, and on how to specify your installation folder at run time. Check the relevant section of the documentation of each module.

When an installation path for a particular module is given in its corresponding input block, it takes precedence over automatic installation folder described above, so that if you already installed a version automatically, it will be ignored in favour of the manually specified one.
