Running on the Amazon EC2 cloud
===============================

.. note::

   This section is work in progress. Let us know about possible corrections/improvements if you try the methods presented here.


Installing and running single jobs
----------------------------------

This is the preferred method for running individual jobs.

First of all, configure and launch an instance with an Ubuntu 16.04 image. For most cosmological applications, we recommend choosing an instance with 16 cores (4 MPI processes threading across 4 cores each) and 32 Gb of RAM (8 Gb per chain). A good choice, following that logic, would be a ``c4.4xlarge`` instance. Set up for it at least 10Gb of storage.

Now install the requisites with

.. code:: bash

   $ sudo apt update && sudo apt install gcc-5 gfortran-5 g++-5 openmpi-bin openmpi-common libopenmpi-dev libopenblas-base liblapack3 liblapack-dev python python-pip

   $ python -m pip install "matplotlib<3" --user  ## this requisite will eventually be removed

   $ python -m pip install mpi4py --user --no-binary :all:

And install **cobaya** (and optionally PolyChord and some cosmology requisites) with

.. code:: bash

   $ python -m pip install cobaya --user

   $ cobaya-install cosmo --packages-path cobaya_packages

.. note::

   If ``cobaya-install cosmo`` fails with a segmentation fault, simply run it again.

Now you are ready to run some samples. Don't forget to mention the external packages folder with ``-p cobaya_packages`` in the command line, or ``packages_path: packages`` in the input file.

As an example, you can just copy the input at :doc:`cosmo_basic_runs`, paste it in a file with ``nano`` and save it to ``planck.yaml``.

To run with ``X`` MPI processes, each creating at most ``Y`` threads (in our recommended configuration, ``X=Y=4``), do

.. code:: bash

   $ mpirun -n X --map-by socket:PE=Y  cobaya-run planck.yaml -p cobaya_packages -o chains/planck
