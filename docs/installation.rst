Installation
============

.. _install_mpi:

Pre-requisites
--------------

The only requisites are:

* Python, version ≥ 2.7
* ``pip``, the Python package manager
* (Optional) an MPI implementation (see below)
* (Optional) the necessary compilers for the cosmological codes and likelihoods
  
To check if you have Python installed, type ``python --version`` in the shell, and you should get ``Python 2.7.[whatever]``. Then, type ``pip`` in the shell, and if you get usage instructions, you are golden. If you don't have any of those two installed, use your system's package manager or contact your local IT service.

Enabling MPI parallelisation is optional but highly recommended: it will allow you to better utilise the size of your cluster. MPI enables inter-process communication, which certain sampler can take advantage of for e.g. achieving a faster convergence of an MCMC proposal distribution, a higher effective acceptance rate in a nested sampler, etc.

First, you need to install an MPI implementation in your system. We recommend `OpenMPI <https://www.open-mpi.org/>`_. Install it using your system's package manager (``libopenmpi`` in Debian-based systems) or contact your local IT service.

Next install Python's `mpi4py` package using `pip`:

.. code:: bash

   $ pip install mpi4py --user

Now try

.. code:: bash

   $ python
   >>> from mpi4py import MPI

If you have a working MPI implementation in your system, this should produce no output. If you don't, the error would look something like ``ImportError: libmpi.so.12: cannot open shared object file``.


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
            

Installing cobaya as a python package
--------------------------------------

This is the recommended method, since it installs all the python dependences automatically.

Download the latest release of cobaya from `the github release page <https://github.com/JesusTorrado/cobaya/releases>`_ or, if you want to use the development version, clone from git in a folder of your choice, say ``/path/to/cosmo/``:

.. code-block:: bash

   $ cd /path/to/cosmo/
   $ git clone https://github.com/JesusTorrado/cobaya.git

This will create a folder called ``cobaya`` inside ``/path/to/cosmo``. Change to it and use ``pip`` to install the package in *editable mode*:

.. code-block:: bash

   $ cd cobaya
   $ pip install --editable . --user

Installing the package in *editable mode* will reflect any changes you make to it. It is not mandatory, but it is a good idea should you want to make any modification in the future.

If everything went well, you should be able to import cobaya in python from anywhere in your directory structure:

.. code-block:: bash

   $ cd
   $ python
   >>> import cobaya       

If you get an error mesage after the ``import`` statement, something went wrong. Check twice the instructions above, try again, or contact us or your local Python guru.
   
cobaya also installs some scripts that should be callable from any folder. If everything went well, if you try to invoke ``cobaya-run`` you should get a message asking you for an input file, instead of a ``command not found`` error.

.. note::

   If you do get a ``command not found`` error, this means that the folder where your local scripts are installed has not been added to your path. In Linux, it should be enough to add the line

   .. code-block:: bash

      export PATH=$PATH:"~/.local/bin/"

   at the end of your ``~/.bashrc`` file, and restart the terminal.

.. note::
      
   As of this version (alpha) there is no public Python package available. In the future, you should be able to install it from the Python Package Index (PyPI) automatically using pip.  


Installing GetDist (only alpha)
-------------------------------

The current version of cobaya is not compatible with the stable version of GetDist, so it cannot be installed as a python requirement, but needs to be cloned from `this github repo <https://github.com/JesusTorrado/getdist>`_ (preferably into ``/path/to/cosmo`` and installed with

.. code-block:: bash

   $ cd /path/to/cosmo/
   $ git clone https://github.com/JesusTorrado/getdist.git
   $ cd getdist
   $ pip install --editable . --user


Cosmology
---------

No cosmological codes or likelihoods are installed by default. They are left to the user for the sake of lightness of this code, and in case they want to use a modified version of them.

To install the usual cosmological codes, see the corresponding *Installation* sections in names_me's documentation of the interfaces for :doc:`CAMB <theory_camb>` and :doc:`CLASS <theory_class>`.

To install the Planck 2015 likelihood, check out the *Installation* section in :doc:`likelihood_planck`.

Uninstalling cobaya (and GetDist, in the alpha)
------------------------------------------------

Simply do, from anywhere

.. code-block:: bash

   $ pip uninstall cobaya getdist

and delete the corresponding folders.

.. note::

   As of this version, the scripts installed in the local ``bin`` folder (in Linux ``~/.local/bin``) are not deleted automatically by the command above. You have to delete them manually -- just get rid of the files there that start with ``cobaya`` or ``GetDist``.
