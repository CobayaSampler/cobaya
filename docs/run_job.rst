Submitting and running jobs
===========================

More time-consuming runs are usually run on a remote cluster, via a job queue. A good configuration for running MCMC chains is often to run 4-6 chains, each using some number of cores (e.g. in cosmology, for OPENMP threading in the cosmology codes, for  a total of 4-24 hours running time). Cobaya has a convenience script to produce, submit and manage job submission scripts that can be adapted for different systems. Once configured you can do::

  cobaya-run-job --queue regular --walltime 12:00:00 [yaml_file].yaml

This produces a job script for your speciifc yaml_file, and then submits it to the queue using default settings for your cluster.

To do this, it loads a template for your job submission script. This template can be specified via a command line argument, e.g. ::

 cobaya-run-job --walltime 12:00:00 --job-template job_script_NERSC [yaml_file].yaml

However it is usually more convenient to set an environment variable on each/any cluster that you use so that the appropriate job script template is automatically used. You can then submit jobs on different clusters with the same commands, without worrying about local differences. To set the environment variable put in your .bashrc (or equivalent)::

 export COBAYA_job_template=/path/to/my_cluster_job_script_template


Job script templates
======================

These are essentially queue submission scripts with variable values replaced by {placeholder}s.
There are also lines to specify default settings for the different cobaya-run-job options. For example for NERSC, the template might be

.. literalinclude:: ../cobaya/grid_tools/script_templates/job_script_NERSC
   :language: shell

Here each word in {} braces is replaced with a value taken (or computed) from your cobaya-run-script arguments.
The ##RUN line specified the actual command. If you run more than one run per job, this may be used multiple times in the generated script file.

The lines starting ## are used to define default settings for jobs, in this case 4 chains each running with 4 cores each (this does not use a complete NERSC node).

The available placeholder variables are:

.. list-table::
   :width: 100%
   :header-rows: 0

   * - JOBNAME
     - name of job, from yaml file name
   * - QUEUE
     - queue name
   * - WALLTIME
     - running time (e.g. 12:00:00 for 12 hrs)
   * - NUMNODES
     - number of nodes
   * - OMP
     - number of OPENMP threads per chain (one chain per mpi process)
   * - CHAINSPERNODE
     - number of chains per node for each run
   * - NUMRUNS
     - number of runs in each job
   * - NUMTASKS
     - total number of chains (NUMMPI * NUMRUNS)
   * - NUMMPI
     - total of MPI processes per run (=total number of chains per run)
   * - MPIPERNODE
     - total number of MPI processes per node (CHAINSPERNODE * NUMRUNS)
   * - PPN
     - total cores per node (CHAINSPERNODE * NUMRUNS * OMP)
   * - NUMSLOTS
     - total number of cores on all nodes (PPN * NUMNODES)
   * - MEM_MB
     - memory requirement
   * - JOBCLASS
     - job class name
   * - ROOTDIR
     - directory of invocation
   * - JOBSCRIPTDIR
     - directory of the generated job submission script file
   * - ONERUN
     - zero if only one run at a time (one yaml or multiple yaml run sequentially)
   * - PROGRAM
     - name of the program to run (cobaya-run) [can be changed by cobaya-run,
       e.g. to change cobaya's optional run arguments]
   * - COMMAND
     - substituted by the command(s) that actually runs the job, calculated from ##RUN

The ##RUN line in the template has the additional placeholders INI and INIBASE,
which are sustituted by the name of the input yaml file, and the base name (without .yaml) respectively.

Optional arguments
===========================

You can change various arguments when submitting jobs, running `cobaya-run-job -h` gives you the details

.. program-output:: cobaya-run-job -h

To set default `yy` for option `xx` in your job script template, add a line::

 ##DEFAULT_xx: yy

Job control
===========================

When you use cobaya-run-job, it stores the job details of this, and all other jobs started from the same directory, in a pickle file in your ./scripts directory (along with the generated and submitted job submission script). This can be used by two additional utility scripts `cobaya-running-jobs` which lists queued jobs, with optional filtering on whether actually running or queued.

Use `cobaay-delete-jobs` to delete a job corresponding to a given input yaml base name, or to delete a range of job ids.
