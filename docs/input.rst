Input and invocation
====================

The input of **cobaya** consists of a collection of Python dictionaries specifying the different parts of the code to use (likelihoods, theory codes and samplers) and which parameters to sample and how. The contents of that dictionary are describe below.

This information if often provided as a text file in the `YAML <https://en.wikipedia.org/wiki/YAML>`_ format. The basics of YAML are better learnt with an example, so go back to :doc:`example` if you have not read it yet. If you are havig trouble making your input in YAML work, take a look at the :ref:`common_yaml_errors` at the bottom of this page.


.. _input_blocks:

Basic input structure
---------------------

There are 5 different input blocks (two of them optional), which can be specified in any order:

- ``likelihood``: contains the likelihoods that are going to be explored, and their respective options — see :doc:`likelihoods`

- ``params``: contains a list of parameters to be *fixed*, *sampled* or *derived*, their priors, LaTeX labels, etc. — see :doc:`params_prior`.

- ``prior``: (optional) contains additional priors to be imposed, whenever they are complex or non-separable — see :ref:`prior_external`.

- ``sampler``: contains the sampler as a single entry, and its options — see :doc:`sampler`.

- ``theory`` (optional): has only one entry, which specifies the theory code with which to compute the observables used by the likelihoods, and options for it. Also, if a ``theory`` is specified, the ``params`` block may contain a ``theory`` sub-block containing the parameters belonging to the theory code — see :doc:`theory` and also :doc:`examples_planck` for a usage example.

The modules specified above (i.e. likelihoods, samplers, theories...) can have any number of options, but you don't need to specify all of them every time you use them: if an option is not specified, its **default** value is used. The default values for each module are described in their respective section of the documentation, and in a ``defaults.yaml`` file in the folder of **cobaya** where that module is defined, e.g. ``cobaya/cobaya/likelihoods/gaussian/defaults.yaml`` for the defaults of the ``gaussian`` likelihood.

In addition, there are some *top level* options (i.e. defined outside any block):

+ ``output_prefix``: determines where the output files are written and/or a prefix for their names — see :ref:`output_shell`.
+ ``path_to_modules``: path where the external modules have been automatically installed — see :doc:`installation_ext`.
+ ``debug``: sets the verbosity level of the output. By default (undefined or ``False``), it produces a rather informative output, reporting on initialization, overall progress and results. If ``True``, it produces a very verbose output (a few lines per sample) that can be used for debugging. You can also set it directly to a particular `integer level of the Python logger <https://docs.python.org/2/library/logging.html#logging-levels>`_, e.g. 40 to produce error output only.
+ ``debug_file``: a file name, with a relative or absolute path if desired, to which to send all logged output. When used, only basic progress info is printed on-screen, and the full debug output (if ``debug: True``) will be sent to this file instead


Running **cobaya**
------------------

You can invoke **cobaya** either from the shell or from a Python script (or notebook).

To run **cobaya** from the shell, use the command ``cobaya-run``, followed by your input file.

.. code:: bash

   $ cobaya-run your_input.yaml

To use MPI, simply run it using the appropriate MPI run script in your system (usually ``mpirun -n [#processes]``).

To run **cobaya** from a Python interpreter, simply do

.. code:: python

    from cobaya.run import run
    updated_info, products = run(your_input)

where your input is a Python dictionary (for how to create one, see :ref:`example_quickstart_interactive`).

To run **cobaya** with MPI in this case, save your script to some file and run ``python your_script.py`` with your MPI run script.


.. _input_cont:

Continuing a sample
-------------------

.. todo::

   Sample continuation is not implemented yet.


.. _common_yaml_errors:

Some common YAML *gotchas*
--------------------------

+ **use colons(+space), not equal signs!** Values are assigned with a ``:``, not a ``=``; e.g. the following input would produce an error:

  .. code:: yaml

     sampler:
       mcmc:
         burn_in = 10   # ERROR: should be 'burn_in: 10'
         max_tries:100  # ERROR: should have a space: 'max_tries: 100'

+ **missing colons!** Each module or parameter definition, even if it is a bare *mention* and does not have options, must end in a colon (which is actually equivalent to writing a null value ``null`` after the colon); e.g. the following input would produce an error:

  .. code:: yaml

     sampler:
       mcmc  # ERROR: no colon!

+ **indentation!** Block indentation must be *coherent*, i.e. everything within the same block must be the same number of spaces to the right; e.g. the following input would produce two errors

  .. code:: yaml

     sampler:
       mcmc:
         burn_in: 10
          max_samples: 100  # ERROR: should be aligned with 'burn_in'

     params:
       mock_a:
         prior:
           min: 0
           max: 1
          latex: \alpha  # ERROR:  should be aligned with 'prior'

  Above, ``max_samples`` should be aligned to ``burn_in``, because both belong into ``mcmc``. In the same way, ``latex`` should be aligned to ``prior``, since both belong into the definition of the parameter ``mock_a``.

.. note::

   For the YAML *connoisseur*, notice that the YAML parser used here has been modified to simplify the input/output notation: it now retains the ordering of parameters and likelihoods (loads mappings as `OrderedDict <https://docs.python.org/2/library/collections.html#ordereddict-examples-and-recipes>`_) and prints arrays as lists.



