Running grids of jobs
================================

Often you need to investigate multiple combinations of parameters and/or likelihoods, or explore a range of different options.
Using Cobaya grids, you can run and manage a set of runs using the grid scripts. This includes tools for submitting jobs to the cluster, managing running jobs, analysing results, and producing tables of results (like the Planck parameter tables).

To create a grid, you need a setting file specifying which combinations of parameters, likelihoods, etc. to use.
The command line to generate the basic structure and files in the ``grid_folder`` directory is::

  cobaya-grid-create grid_folder [my_file]

where ``[my_file]`` is either a .py python setting file or a .yaml grid description.
This will create ``grid_folder`` if it does not exist, and generate a set of .yaml files for running each of the runs in the grid.
There is a simple generic `python example <https://github.com/CobayaSampler/cobaya/blob/master/tests/simple_grid.py>`_  and a cosmology `yaml example <https://github.com/CobayaSampler/cobaya/blob/master/tests/test_cosmo_grid.yaml>`_  which combines single parameter variations each with two different likelihoods.

Once the grid is created, you can check the list of runs included using::

  cobaya-grid-list grid_folder

To actually submit and run the jobs, you'll need to have a :doc:`job script template <run_job>` configured for your cluster. You can check which jobs will be submitted using::

  cobaya-grid-run grid_folder --dryrun

Simply remove the ``--dryrun`` to actually submit the jobs to run each of the items in the grid. Most of grid scripts have optional parameters to filter the grid to only run on specific subsets of items; use the ``-h`` option to see the full help.

You can use ``cobaya-running-jobs grid_folder`` to and monitor which jobs are queued and running, and ``cobaya-delete-jobs grid_folder`` to cancel jobs based on various name filters.

After the main samples are generated, if you have ``importance_runs`` set you can do the corresponding importance sampling on the generated chains using::

  cobaya-grid-run grid_folder --importance

The grid also generates input files for minimization rather than sampling runs. If you also want best fits, run::

 cobaya-grid-run grid_folder --minimize

Any custom settings for minimization come from the ``minimize_defaults`` dictionary in the input grid settings.

For best-fits from importance sampled grid combinations, run::

  cobaya-grid-run grid_folder --importance_minimize

For any run that is expected to be fast, you can use ``--noqueue`` to run each item directly rather than using a queue submission.

Analysing grid results
================================

While jobs are still running, you can use::

  cobaya-grid-converge grid_folder --checkpoint

to show the convergence of each chain from the current checkpoint file. Filter to show on running jobs with ``--running``, or conversely ``--not-running``. If runs stop before they are converged, e.g. due to wall time limits, you can use::

  cobaya-grid-run grid_folder --checkpoint_run

to re-start the finished runs that are not converged. ``--checkpoint_run`` has an optional parameter to an R-1 convergence value so that only chains with worse convergence than that are rerun.

To see parameter constraints, and convergence statistics using the written chain files, use::

 cobaya-grid-getdist grid_folder --burn_remove 0.3

This will run GetDist on all the chains in the folder, removing the first 30% of each chain as burn in. You can use this while chains are still running, and incrementally update later by adding the ``--update_only`` switch (which only re-analyses chains which have changed). GetDist text file outputs are stored under a ``/dist`` folder in each subfolder of the grid results. To view GetDist-generated convergence numbers use::

 cobaya-grid-converge grid_folder

Tables of grid results
================================

After running ``cobaya-grid-getdist``, you can combine results into a single PDF document (or latex file) containing tables of all the results::

  cobaya-grid-tables grid_folder output_file --limit 2

This example will output tables containing 95% confidence limits by default, use ``--limit 1`` to get 68% confidence tables. Alternatively use ``--all_limits`` to include all. By default the script generates a latex file and then compiles it with ``pdflatex``. Use the ``--forpaper`` option to only generate latex.

There are a number of options to customize the result, compare results and give shift significances, e.g.::

  cobaya-grid-tables grid_folder tables/baseline_params_table_95pc --limit 2 --converge 0.1 --musthave_data NPIPE lowl lowE --header_tex tableHeader.tex --skip_group nonbbn --skip_data JLA reion BK18

If you want a latex table to compare different parameter and data combinations, you can use the ``cobaya-grid-tables-compare`` command, see ``-h`` for options.

Exporting grid results
================================

To copy a grid for distribution, without including unwanted files, use::

  cobaya-grid-copy grid_folder grid_folder_export.zip

Add the ``--dist`` option to include GetDist outputs, or ``--remove_burn_fraction 0.3`` to delete the first 30% of each chain file as burn in. You can also copy to a folder rather than .zip.

To extract a set of files from a grid, e.g. all GetDist ``.margestats`` table outputs and ``.covmats``, use e.g.::

  cobaya-grid-extract grid_folder output_dir .margestats .covmat

The ``cobaya-grid-cleanup`` script can be used to delete items in a grid_folder, e.g. to free space, delete incorrect results before a re-run, etc.

Grid script parameters
================================

.. program-output:: cobaya-grid-create -h

.. program-output:: cobaya-grid-run -h

.. program-output:: cobaya-grid-converge -h

.. program-output:: cobaya-grid-getdist -h

.. program-output:: cobaya-grid-tables -h

.. program-output:: cobaya-grid-tables-compare -h

.. program-output:: cobaya-grid-copy -h

.. program-output:: cobaya-grid-extract -h

.. program-output:: cobaya-grid-list -h

.. program-output:: cobaya-grid-cleanup -h
