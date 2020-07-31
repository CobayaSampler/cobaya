Installing cobaya
=================

Pre-requisites
--------------

The only pre-requisites are **Python** (version ≥ 3.6) and the Python package manager **pip** (version ≥ 20.0).

.. warning::

   Python 2 is no longer supported. Please use Python 3.

   In some systems, the Python 3 command may be ``python3`` instead of ``python``. In this documentation, the shell command ``python`` always means Python 3.

To check if you have Python installed, type ``python --version`` in the shell, and you should get ``Python 3.[whatever]``. Then, type ``python -m pip --version`` in the shell, and see if you get a proper version line starting with ``pip 20.0.0 [...]`` or a higher version. If an older version is shown, please update pip with ``python -m pip install pip --upgrade``. If either Python 3 is not installed, or the ``pip`` version check produces a ``no module named pip`` error, use your system's package manager or contact your local IT service.

.. note::

   In the following, commands to be run in the shell are displayed here with a leading ``$``. You do not have to type it.

.. note::

   Some of cobaya components (likelihood, Boltzmann codes, samplers) consist only of an interface to some external code or data that will need to be installed separately (see :doc:`installation_cosmo`).


.. _install_mpi:

MPI parallelization (optional but encouraged!)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enabling MPI parallelization is optional but highly recommended: it will allow you to better utilise the size of your CPU or cluster. MPI enables inter-process communication, of which many samplers can take advantage, e.g. for achieving a faster convergence of an MCMC proposal distribution, or a higher effective acceptance rate in a nested sampler.

First, you need to install an MPI implementation in your system, or load the corresponding module in a cluster with ``module load`` (it will appear as ``openmpi``, ``mpich`` or ``pmi``; check your cluster's usage guidelines).

For your own laptop we recommend `OpenMPI <https://www.open-mpi.org/>`_. Install it using your system's package manager (``sudo apt install libopenmpi`` in Debian-based systems) or contact your local IT service.

Second, you need to install the Python wrapper for MPI, ``mpi4py``, with version ``>= 3.0.0``.

.. code:: bash

   $ python -m pip install "mpi4py>=3" --upgrade --no-binary :all:

.. note::

   If you are using Anaconda, do instead

   .. code:: bash

      $ conda install -c [repo] mpi4py

   where ``[repo]`` must be either ``conda-forge`` (if you are using GNU compilers) or ``intel``.

To test the installation, run in a terminal

.. code:: bash

   $ mpirun -n 2 python -c "from mpi4py import MPI, __version__; print(__version__ if MPI.COMM_WORLD.Get_rank() else '')"

(You may need to substitute ``mpirun`` for ``srun`` in certain clusters.)

This should print the version of ``mpi4py``, e.g. ``3.0.0``. If it prints a version smaller than 3, doesn't print anything, or fails with an error similar to ``ImportError: libmpi.so.12``, make sure that you have installed/loaded an MPI implementation and repeat the installation, or ask your local IT service for help.


.. _install:

Installing and updating cobaya
------------------------------

To install **cobaya** or upgrade it to the latest release, simply type in a terminal

.. code:: bash

   $ python -m pip install cobaya[gui] --upgrade

For a **cluster** install, you may want to remove the ``[gui]`` to avoid errors due to non-essential dependencies.

To go on installing **cosmological requisites**, see :doc:`installation_cosmo`.

.. warning::

   In general, use ``python -m pip`` (or ``conda``) **instead of cloning directly from the github repo**: there is where development happens, and you may find bugs and features just half-finished.

   Unless, of course, that you want to help us develop **cobaya**. In that case, take a look at :ref:`install_devel`.


.. _install_check:

Making sure that cobaya is installed
------------------------------------

If everything went well, you should be able to import **cobaya** in Python from anywhere in your directory structure:

.. code-block:: bash

   $ python -c "import cobaya"

If you get an error message, something went wrong. Check twice the instructions above, try again, or contact us or your local Python guru.

**cobaya** also installs some shell scripts. If everything went well, if you try to run in the shell ``cobaya-run``, you should get a message asking you for an input file, instead of a ``command not found`` error.

.. note::

   If you do get a ``command not found`` error, this means that the folder where your local scripts are installed has not been added to your path.

   To solve this on unix-based machines, look for the ``cobaya-run`` script from your ``home`` and ``scratch`` folders with

   .. code-block:: bash

      $ find `pwd` -iname cobaya-run -printf %h\\n

   This should print the location of the script, e.g. ``/home/you/.local/bin``. Add

   .. code-block:: bash

      $ export PATH="/home/you/.local/bin":$PATH

   at the end of your ``~/.bashrc`` file, and restart the terminal or do ``source ~/.bashrc``. Alternatively, simply add that line to your cluster scripts just before calling ``cobaya-run``.


Uninstalling cobaya
-------------------

Simply do, from anywhere

.. code-block:: bash

   $ python -m pip uninstall cobaya

.. note::

   If you installed **cobaya** in *development mode* (see below), you will also have to delete its folder manually, as well as the scripts installed in the local ``bin`` folder (see note above about how to find it).


Installation troubleshooting
----------------------------

.. note::

   This section will be filled with the most common problems that our users encounter, so if you followed the instructions above and still something failed (or if you think that the instructions were not clear enough), don't hesitate to contact us!


.. _install_openblas:

Low performance: install OpenBLAS (or MKL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BLAS is a collection of algorithms for linear algebra computations. There will most likely be a BLAS library installed already in your system. It is recommended to make sure that it is an efficient one, preferably the highly-optimized OpenBLAS or MKL.

Conda installations should include BLAS by default. On other installations check whether ``numpy`` is actually using OpenBLAS or MKL, do

.. code:: bash

   $ python -c "from numpy import show_config; show_config()" | grep 'mkl\|openblas_info' -A 1

Check that it prints a list of libraries and not a ``NOT AVAILABLE`` below *at least one* of ``openblas_info`` or ``blas_mkl_info``.

If you just got ``NOT AVAILABLE``\ 's, load the necessary libraries with ``module load`` if you are in a cluster, or install OpenBlas or MKL.

To check if OpenBLAS is installed, in Debian-like systems, type

.. code:: bash

   $ dpkg -s libopenblas-base | grep Status

The output should end in ``install ok installed``. If you don't have it installed, in a Debian-like system, type ``sudo apt install libopenblas-base`` or ask your local IT service.

You may need to re-install ``numpy`` after loading/installing OpenBLAS.

To check that this worked correctly, run the following one-liner with the same Python that Cobaya is using, and check that ``top`` reports more than 100% CPU usage:

    .. code:: python

       import numpy as np ; (lambda x: x.dot(x))((lambda n: np.reshape(np.random.random(n**2), (n,n)))(10000))


Installing cobaya in development mode
-------------------------------------

Use this method if you want to make modifications to the code, either for yourself, or to collaborate with us by implementing a new feature.

.. note::

   Notice that you don't need to modify **cobaya**'s source to use your own priors, likelihoods, etc. Take a look at the documentation of the components that you would like to modify to check if can do that in an easier way.


.. _install_devel:

Method 1: Using ``git`` (recommended!)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To download and install **cobaya** in *development mode* you will need ``git`` (`learn how to use git <https://git-scm.com/book/en/v2>`_). Type ``git`` in the shell and check that you get usage instructions instead of a ``command not found`` error. In the later case, in a Debian-like system, install it with a ``sudo apt install git``.

The recommended way is to get a `GitHub <https://github.com>`_ user and `fork the cobaya repo <https://help.github.com/articles/fork-a-repo/>`_. Then clone your fork and install it as a Python package in *development mode* (i.e. your changes to the code will have an immediate effect, without needing to update the Python package):

.. code:: bash

   $ git clone https://YOUR_USERNAME@github.com/YOUR_USERNAME/cobaya.git
   $ python -m pip install --editable cobaya[test,gui] --upgrade

Here ``cobaya[test,gui]`` should include the square brackets. Remove ``,gui`` if desired to avoid unnecessary dependencies.

Alternatively, you can clone from the official **cobaya** repo (but this way you won't be able to upload your changes!).

.. code:: bash

   $ git clone https://github.com/CobayaSampler/cobaya.git
   $ python -m pip install --editable cobaya[test,gui] --upgrade

In any of both cases, this puts you in the last commit of **cobaya**, and install the requisites for both running and testing (to ignore the testing requisites, remove ``[test]`` from the commands above). If you want to start from the last release, say version 1.0, do, from the cobaya folder,

.. code:: bash

   $ git checkout v1.0

Finally, take a look at :ref:`install_check`.


Method 2: Simplest, no ``git`` (not recommended!)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   This method is not recommended: you will not be able to keep track of your changes to the code! We really encourage you to use ``git`` (see method 1).

Download the latest release (the one on top) from **cobaya**'s `GitHub Releases page <https://github.com/CobayaSampler/cobaya/releases>`_. Decompress it in some folder, e.g. ``/path/to/cobaya/``, and install it as a python package:

.. code-block:: bash

   $ cd /path/to/cobaya/
   $ python -m pip install --editable cobaya

Finally, take a look at :ref:`install_check`.
