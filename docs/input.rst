Input and invocation
====================

The input to **cobaya** consists of a Python dictionary specifying the different parts of the code to use (likelihoods, theory codes and samplers), which parameters to sample and how, and various options. The contents of that dictionary are described below.

The input dictionary information if often provided as a text file in the `YAML <https://en.wikipedia.org/wiki/YAML>`_ format. The basics of YAML are better learnt with an example, so go back to :doc:`example` if you have not read it yet. If you are having trouble making your input in YAML work, take a look at the :ref:`common_yaml_errors` at the bottom of this page.


.. _input_blocks:

Basic input structure
---------------------

There are 5 different input blocks (two of them optional), which can be specified in any order:

- ``likelihood``: contains the likelihoods that are going to be explored, and their respective options — see :doc:`likelihoods`

- ``params``: contains a list of parameters to be *fixed*, *sampled* or *derived*, their priors, LaTeX labels, etc. — see :doc:`params_prior`.

- ``prior``: (optional) contains additional priors to be imposed, whenever they are complex or non-separable — see :ref:`prior_external`.

- ``sampler``: contains the sampler as a single entry, and its options — see :doc:`sampler`.

- ``theory`` (optional): specifies the theory code(s) with which to compute the observables used by the likelihoods, and their options.

The *components* specified above (i.e. likelihoods, samplers, theories...) can have any number of options, but you don't need to specify all of them every time you use them: if an option is not specified, its **default** value is used. The default values for each component are described in their respective section of the documentation, and in a ``[likelihood_name].yaml`` file in the folder of **cobaya** where that component is defined, e.g. ``cobaya/cobaya/likelihoods/gaussian/gaussian.yaml`` for the defaults of the ``gaussian`` likelihood.

In addition, there are some *top level* options (i.e. defined outside any block):

+ ``output``: determines where the output files are written and/or a prefix for their names — see :ref:`output_shell`.
+ ``packages_path``: path where the external packages have been automatically installed — see :doc:`installation_cosmo`.
+ ``debug``: sets the verbosity level of the output. By default (undefined or ``False``), it produces a rather informative output, reporting on initialization, overall progress and results. If ``True``, it produces a very verbose output (a few lines per sample) that can be used for debugging. You can also set it directly to a particular `integer level of the Python logger <https://docs.python.org/2/library/logging.html#logging-levels>`_, e.g. 40 to produce error output only (alternatively, ``cobaya-run`` can take the flag ``--debug`` to produce debug output, that you can pipe to a file with ``>file``).
+ ``debug_file``: a file name, with a relative or absolute path if desired, to which to send all logged output. When used, only basic progress info is printed on-screen, and the full debug output (if ``debug: True``) will be sent to this file instead


Running **cobaya**
------------------

You can invoke **cobaya** either from the shell or from a Python script (or notebook).

To run **cobaya** from the shell, use the command ``cobaya-run``, followed by your input file.

.. code:: bash

   $ cobaya-run your_input.yaml

.. note::

   To use **MPI**, simply run it using the appropriate MPI run script in your system, e.g.

   .. code:: bash

      $ mpirun -n [#processes] cobaya-run your_input.yaml

   If you get an error of the kind ``mpirun was unable to find the specified executable file [...]``, you will need to specify the full path to the ``cobaya-run`` script, e.g.

   .. code:: bash

      $ mpirun -n [#processes] $HOME/.local/bin/cobaya-run your_input.yaml

   .. warning::

      In rare occasions, when ``KeyboardInterrupt`` is raised twice in a row within a small interval, i.e. when :kbd:`Control-c` is hit twice really fast, secondary processes may not die, and need to be killed manually.

      If you notice secondary process not dying by themselves in any other circumstance, please contact us, including as much information on the run as possible.


To run **cobaya** from a Python interpreter, simply do

.. code:: python

    from cobaya.run import run
    updated_info, sampler = run(your_input)

where ``your_input`` is a Python dictionary (for how to create one, see :ref:`example_quickstart_interactive`).

To run **cobaya** with MPI in this case, save your script to some file and run ``python your_script.py`` with your MPI run script.


.. _input_resume:

Resuming or overwriting an existing run
------------------------------------------

If the input refers to output that already exists, **cobaya** will, by default, let you know and produce an error.

To overwrite previous results (**use with care!**), either:

* Set ``force: True`` in the input.
* Invoke ``cobaya-run`` with a ``-f`` (or ``--force``) flag.

.. warning::

   Do not overwrite an MCMC sample with a PolyChord one using this (or the other way around); delete it by hand before re-running. This will be fixed in a future release.

If instead you would like to **resume a previous sample**, either:

* Set ``resume: True`` in the input.
* Invoke ``cobaya-run`` with a ``-r`` (or ``--resume``) flag.

In this case, the new input will be compared to the existing one, and an error will be raised if they are not compatible, mentioning the first part of the input that was found to be inconsistent.

.. note::

   Differences in options that do not affect the statistics will be ignored (e.g. parameter labels). In this case, the new ones will be used.

.. note::

   Resuming by invoking ``run`` interactively (inside a Python notebook/script), it is *safer* to pass it the **updated** info of the previous run, instead of the one passed to the first call (otherwise, e.g. version checks are not possible).

An alternative way of resuming a sample *from the command line* is passing, instead of a ``yaml`` file, the ``output`` of an existing one:

.. code:: bash

   $ cobaya-run input.yaml    # writes into 'output: chains/gauss'
   $ cobaya-run chains/gauss  # continues the previous one; no need for -r!!!

.. note::

   if ``output`` ends with a directory separator (``/``) this has to be included in the resuming call too!


.. _common_yaml_errors:

Some common YAML *gotchas*
--------------------------

+ **specify infinities with** ``.inf``

  .. code:: yaml

     a: .inf  # this produces the *number* Infinity
     b: +.inf  # this produces the *number* Infinity
     c: -.inf  # this produces the *number* -Infinity
     d: inf  # this produces the *string* 'inf' (won't fail immediately)


+ **use colons(+space), not equal signs!** Values are assigned with a ``:``, not a ``=``; e.g. the following input would produce an error:

  .. code:: yaml

     sampler:
       mcmc:
         burn_in = 10   # ERROR: should be 'burn_in: 10'
         max_tries:100  # ERROR: should have a space: 'max_tries: 100'

+ **missing colons!** Each component or parameter definition, even if it is a bare *mention* and does not have options, must end in a colon (which is actually equivalent to writing a null value ``null`` after the colon); e.g. the following input would produce an error:

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
       a:
         prior:
           min: 0
           max: 1
          latex: \alpha  # ERROR:  should be aligned with 'prior'

  Above, ``max_samples`` should be aligned to ``burn_in``, because both belong into ``mcmc``. In the same way, ``latex`` should be aligned to ``prior``, since both belong into the definition of the parameter ``a``.

.. note::

   For the YAML *connoisseur*, notice that the YAML parser used here has been modified to simplify the input/output notation: it now retains the ordering of parameters and likelihoods and prints arrays as lists.



