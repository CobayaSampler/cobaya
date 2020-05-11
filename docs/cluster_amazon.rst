Running on the Amazon EC2 cloud
===============================

.. note::

   This section is work in progress. Let us know about possible corrections/improvements if you try the methods presented here.


Installing and running single jobs
----------------------------------

This is the preferred method for running individual jobs.

First of all, configure and launch a Linux image. For most cosmological applications, we recommend choosing an Ubuntu 18.04 instance with about 16 cores (4 MPI processes threading across 4 cores each) and 32 Gb of RAM (8 Gb per chain). A good choice, following that logic, would be a ``c5d.4xlarge`` (compute optimized) instance. Set up for it at least 10Gb of storage.

Now install the requisites with

.. code:: bash

   $ sudo apt update && sudo apt install gcc gfortran g++ openmpi-bin openmpi-common libopenmpi-dev libopenblas-base liblapack3 liblapack-dev
   $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
   $ bash miniconda.sh -b -p $HOME/miniconda
   $ export PATH="$HOME/miniconda/bin:$PATH"
   $ conda config --set always_yes yes --set changeps1 no
   $ conda create -q -n cobaya-env python=3.7 scipy matplotlib cython PyYAML pytest pytest-forked flaky
   $ source activate cobaya-env
   $ pip install mpi4py

And install **cobaya** (and optionally PolyChord and some cosmology requisites) with

.. code:: bash

   $ pip install cobaya

   $ cobaya-install cosmo --packages-path cobaya_packages

Now you are ready to run some samples.

As an example, you can just copy the input at :doc:`cosmo_basic_runs`, paste it in a file with ``nano`` and save it to ``planck.yaml``.

To run with ``X`` MPI processes, each creating at most ``Y`` threads (in our recommended configuration, ``X=Y=4``), do

.. code:: bash

   $ mpirun -n X --map-by socket:PE=Y  cobaya-run planck.yaml -p cobaya_packages -o chains/planck
