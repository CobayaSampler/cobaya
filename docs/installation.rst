Installing cobaya
=================

Pre-requisites
--------------

The only pre-requisites are **Python** (version ≥ 2.7) and the Python package manager **pip** (external modules may have additional dependencies).

To check if you have Python installed, type ``python --version`` in the shell, and you should get ``Python 2.7.[whatever]``. Then, type ``pip`` in the shell, and if you get usage instructions instead of a ``command not found`` message, you are golden. If you don't have any of those two installed, use your system's package manager or contact your local IT service.

.. _install_mpi:

Optional: MPI parallelisation
-----------------------------

Enabling MPI parallelisation is optional but highly recommended: it will allow you to better utilise the size of your cluster. MPI enables inter-process communication, of which many samplers can take advantage, e.g. for achieving a faster convergence of an MCMC proposal distribution, or a higher acceptance rate in a nested sampler.

First, you need to install an MPI implementation in your system. We recommend `OpenMPI <https://www.open-mpi.org/>`_. Install it using your system's package manager (``sudo apt install libopenmpi`` in Debian-based systems) or contact your local IT service.

Next install Python's ``mpi4py``:

.. code:: bash

   $ pip install mpi4py --upgrade

Now try

.. code:: bash

   $ python -c "from mpi4py import MPI"

If you have a working MPI implementation in your system, this should produce no output. If you don't, the error would look something like ``ImportError: libmpi.so.12: cannot open shared object file``.


Optional: make cobaya faster with OpenBLAS
------------------------------------------

BLAS is a collection of algorithms for linear algebra computations. There will most likely be a BLAS library installed already in your system. It is recommended to make sure that it is an efficient one, preferably the highly-parallelised OpenBLAS.

To check if OpenBLAS is installed, in Debian-like systems, type

.. code:: bash

   $ dpkg -s libopenblas-base | grep Status

The output should end in ``install ok installed``. If you don't have it installed, in a Debian-like system, type ``sudo apt install libopenblas-base`` or ask your local IT service.


Installing and updating cobaya
------------------------------

.. warning::
      
   Fro now, during beta testing, there is an additional pre-requisite: a ``git`` installation. Type ``git`` in the shell and check that you get usage instructions instead of a ``command not found``. In the latter case, in a Debian-like system, install it with a ``sudo apt install git``.


To install **cobaya** or upgrade it to the last release, simply type in a terminal

.. code:: bash

   $ pip install git+https://github.com/JesusTorrado/getdist/\#egg=getdist --upgrade
   $ pip install cobaya --upgrade


.. _install_check:
   
Make sure that cobaya is installed
----------------------------------   
   
If everything went well, you should be able to import **cobaya** in Python from anywhere in your directory structure:

.. code-block:: bash

   $ python -c "import cobaya"

If you get an error message, something went wrong. Check twice the instructions above, try again, or contact us or your local Python guru.

**cobaya** also installs some shell scripts. If everything went well, if you try to run in the shell ``cobaya-run``, you should get a message asking you for an input file, instead of a ``command not found`` error.

.. note::

   If you do get a ``command not found`` error, this means that the folder where your local scripts are installed has not been added to your path. In Linux, it should be enough to add the line

   .. code-block:: bash

      export PATH=$PATH:"~/.local/bin/"

   at the end of your ``~/.bashrc`` file, and restart the terminal (or do ``source ~/.bashrc``).


Troubleshooting
---------------

.. note::

   This section will be filled with the most common problems that our users encounter, so if you followed the instructions above and still something failed (or if you think that the instructions were not clear enough), don't hesitate to contact us!


Installing cobaya in development mode
-------------------------------------

Use this method if you want to make modifications to the code, either for yourself, or to collaborate with us by implementing a new feature.

.. note::

   Notice that you don't need to modify **cobaya**'s source to use your own priors, likelihoods, etc. Take a look at the documentation of the modules that you would like to modify.


Method 1: Using ``git`` (recommended!)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To download and install **cobaya** in *development mode* you will need ``git`` (`learn how to use git <https://git-scm.com/book/en/v2>`_). Type ``git`` in the shell and check that you get usage instructions instead of a ``command not found`` error. In the later case, in a Debian-like system, install it with a ``sudo apt install git``.

The recommended way is to get a `GitHub <https://github.com>`_ user and `fork the cobaya repo <https://help.github.com/articles/fork-a-repo/>`_. Then clone you fork and install it as a Python package in *development mode* (i.e. your changes to the code will have an immediate effect, without needing to update the Python package):

.. code:: bash

   $ git clone https://YOUR_USERNAME@github.com/YOUR_USERNAME/cobaya.git
   $ pip install --editable cobaya --upgrade

Alternatively, you can clone from the official **cobaya** repo (but this way you won't be able to upload your changes!).

.. code:: bash

   $ git clone https://github.com/JesusTorrado/cobaya.git
   $ pip install --editable cobaya --upgrade

In any of both cases, this puts you in the last commit of **cobaya**. If you want to start from the last release, say version 1.0, do, from the cobaya folder,

.. code:: bash

   $ git checkout v1.0

Finally, install **GetDist**:
   
.. code:: bash

   $ pip install git+https://github.com/JesusTorrado/getdist/\#egg=getdist --upgrade

and finally :ref:`install_check`.


Method 2: Simplest, no ``git`` (not recommended!)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   This method is not recommended: you will not be able to keep track of your changes to the code! We really encourage you to use ``git`` (see method 1).

Download the latest release (the one on top) from **cobaya**'s `GitHub Releases page <https://github.com/JesusTorrado/cobaya/releases>`_. Decompress it in some folder, e.g. ``/path/to/cobaya/``, and install it as a python package:

.. code-block:: bash

   $ cd /path/to/cobaya/
   $ pip install --editable cobaya

Then install **GetDist**:

.. code:: bash

      $ wget https://github.com/JesusTorrado/getdist/archive/master.zip
      $ unzip master.zip ; rm master.zip
      $ mv getdist-master getdist
      $ pip install getdist
      $ rm -rf getdist

Finally, :ref:`install_check`.  


Uninstalling cobaya
-------------------

Simply do, from anywhere

.. code-block:: bash

   $ pip uninstall cobaya getdist

.. note::

   If you installed **cobaya** in development mode, you will also have to delete its folder manually, as well as the scripts installed in the local ``bin`` folder (the files starting with ``cobaya`` and ``GetDist``, which in Linux should be in ``~/.local/bin``).
