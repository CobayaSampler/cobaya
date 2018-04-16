Advanced example
================

In this example, we will see how to sample from priors and likelihoods given as Python functions, and how to dynamically define new parameters. This time, we will start from the interpreter and then learn how to create a pure ``yaml`` input file with the same information.


.. _example_advanced_interactive:

From a Python interpreter
-------------------------

Our likelihood will be a gaussian ring centred at 0 with radius 1. We define it with the following Python function and add it to the information dictionary like this:

.. code:: python

    import numpy as np
    from scipy import stats

    def gauss_ring_logp(x, y):
        return stats.norm.logpdf(np.sqrt(x**2+y**2), loc=1, scale=0.2)

    info = {"likelihood": {"ring": gauss_ring_logp}}


.. note::

   NB: external likelihood and priors (as well as internal ones) must return **log**-probabilities.


**cobaya** will automatically recognise ``x`` and ``y`` (or whatever parameter names of your choice) as the input parameters of that likelihood, which we have named ``ring``. Let's define a prior for them:

.. code:: python

    from collections import OrderedDict as odict

    info["params"] = odict([
        ["x", {"prior": {"min": -2, "max": 2}, "ref": 1, "proposal": 0.2}],
        ["y", {"prior": {"min": -2, "max": 2}, "ref": 0, "proposal": 0.2}]])


Now, let's assume that we want to track the radius of the ring, whose posterior will be approximately gaussian, and the angle, whose posterior will be uniform. We can define them as function of known input parameters:

.. code:: python

    r = lambda x,y: np.sqrt(x**2+y**2)
    theta = lambda x,y: np.arctan(y/x)

    info["params"]["r"] = {"derived": r}
    info["params"]["theta"] = {"derived": theta, "latex": r"\theta"}


Now, we add the sampler information and run. Notice the high number of samples requested for just two dimensions, in order to map the curving posterior accurately:

.. code:: python

    info["sampler"] = {
        "mcmc": {"burn_in": 500, "max_samples": 10000}}

    from cobaya.run import run
    updated_info, products = run(info)


And now we plot the posterior for ``x``, ``y``, the radius and the angle:

.. code:: python

    %matplotlib inline
    from getdist.mcsamples import loadCobayaSamples
    import getdist.plots as gdplt

    gdsamples = loadCobayaSamples(updated_info, products["sample"])
    gdplot = gdplt.getSubplotPlotter(width_inch=5)
    gdplot.triangle_plot(gdsamples, ["x", "y"], filled=True)
    gdplot = gdplt.getSubplotPlotter(width_inch=5)
    gdplot.plots_1d(gdsamples, ["r", "theta"], nx=2)


.. image:: img/example_adv_ring.png
.. image:: img/example_adv_r_theta.png


Now let's assume that we are only interested in the region ``x>y``. We can add this constraint as an *external prior*, in a similar way the external likelihood was added. The logprior for this can be added simply as:

.. code:: python

    info["prior"] = {"xGTy": lambda x,y: np.log(x>y)}

(Notice that in Python the numerical value of ``True`` and ``False`` are respectively 0 and 1. This will print a single *Warning*, since :math:`log(0)` is not finite, but **cobaya** has no problem dealing with infinities.)

Let's run with the same configuration and analyse the output:

.. code:: python

    updated_info_xGTy, products_xGTy = run(info)

    gdsamples_xGTy = loadCobayaSamples(
        updated_info_xGTy, products_xGTy["sample"])
    gdplot = gdplt.getSubplotPlotter(width_inch=5)
    gdplot.triangle_plot(gdsamples_xGTy, ["x", "y"], filled=True)


.. image:: img/example_adv_half.png


.. _example_advanced_rtheta:

Alternative: sampling from ``r`` and ``theta`` directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The posterior on the radius and the angle is a gaussian times a uniform, much simpler than that on ``x`` and ``y``. So we should probably sample on ``r`` and ``theta`` instead, where we would get a more accurate result with the same number of samples, since now we don't have the problem of having to go around the ring.

This can be done in a simple way at the level of the parameters, i.e. without needing to modify the parameters that the likelihood takes, as explained in :ref:`repar`. In essence:

* We give a prior to the parameters over which we want to sample, here ``r`` and ``theta``, and give them the property ``drop: True`` if they are not understood by the likelihood.
* We define the parameters taken by the likelihood, here ``x`` and ``y``, by a function of sampled and fixed parameters, here ``r`` and ``theta``.
* If we want to have those likelihood parameters showing up as derived, we add a dummy derived parameter for each of them, here ``xprime`` and ``yprime``, with trivial definitions in terms of the original likelihood parameters.

.. code:: yaml

    from copy import deepcopy
    info_rtheta = deepcopy(info)
    info_rtheta["params"] = odict([
        ["r", {"prior": {"min": 0, "max": 2}, "ref": 1,
               "proposal": 0.5, "drop": True}],
        ["theta", {"prior": {"min": -0.75*np.pi, "max": np.pi/4}, "ref": 0,
                   "proposal": 0.5, "latex": r"\theta", "drop": True}],
        ["x", "lambda r,theta: r*np.cos(theta)"],
        ["y", "lambda r,theta: r*np.sin(theta)"],
        ["xprime", {"derived": "lambda x: x"}],
        ["yprime", {"derived": "lambda y: y"}]])
    # The x>y condition is already incorporated in the prior of theta
    info_rtheta["prior"].pop("xGTy")


.. _example_advanced_shell:

From the shell
--------------

To run the example above in from the shell, we could just save all the Python code above in a text file and run it with ``python [file_name]``. To get the sampling results as text output, we would add to the ``info`` dictionary some ``output_prefix``, e.g. ``info["output_prefix"] = "chains/ring"``.

But there a small complication: **cobaya** would fail at the time of dumping a copy of the information dictionary, since there is no way to dump a pure Python function to pure-text ``yaml`` in a reproducible manner. To solve that, for functions that can be written in a single line, we simply write it ``lambda`` form and wrap it in quotation marks, e.g. ``r = "lambda x,y: np.sqrt(x**2+y**2)"``. Inside this lambdas, you can use ``np`` for ``numpy`` and ``stats`` for ``scipy.stats``.

Longer functions must be saved to a separate file and imported on the fly. In the example above, let's assume that we have saved the definition of the gaussian ring likelihood (which could actually be written in a single line anyway), to a file called ``my_likelihood`` in the same folder as the Python script. In that case, we would load the likelihood as

.. code::

    # Notice the use of single vs double quotes
    info = {"likelihood": {"ring": "import_module('my_likelihood').ring"}}


With those changes, we would be able to run out Python script from the shell (with MPI, if desired) and have the chains saved where requested. We could also have incorporated those text definitions into a ``yaml`` file, that we could call with ``cobaya-run``:

.. code:: yaml

    likelihood:
      ring: import_module('my_likelihood').gauss_ring_logp

    params:
      x:
        prior: {max: 2, min: -2}
        ref: 1
        proposal: 0.2
      y:
        prior: {max: 2, min: -2}
        ref: 0
        proposal: 0.2
      r:
        derived: 'lambda x,y: np.sqrt(x**2+y**2)'
      theta:
        derived: 'lambda x,y: np.arctan(y/x)'
        latex: \theta

    prior:
      xGTy: 'lambda x,y: np.log(x>y)'

    sampler:
      mcmc:
        burn_in: 500
        learn_proposal: true
        max_samples: 10000

    output_prefix: chains/ring


.. note::

   Notice that we keep the quotes around the definition of the ``lambda`` functions, or ``yaml`` would get confused by the ``:``.
