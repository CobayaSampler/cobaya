``evaluate`` sampler
====================

This is a *dummy* sampler that just evaluates the likelihood at a *reference* point. You can use it to test your likelihoods (take a look too at the :doc:`model wrapper <cosmo_model>` for a similar but more interactive tool).

To use it, simply make the ``sampler`` block:

.. code-block:: yaml

   sampler:
     evaluate:
       # Optional: override parameter values
       override:
         # param: value

The posterior will be evaluated at a point sampled from the *reference* pdf (which may be a fixed value) or from the prior if there is no reference. Values passed through ``evaluate:override`` will take precedence. For example:

.. code-block:: yaml

   params:
     a:
       prior:
         min: -1
         max:  1
       ref: 0.5
     b:
       prior:
         min: -1
         max:  1
       ref:
         dist: norm
         loc: 0
         scale: 0.1
     c:
       prior:
         min: -1
         max:  1
     d:
       prior:
         min: -1
         max:  1
       ref: 0.4

   sampler:
     evaluate:
       override:
         d: 0.2


In this case, the posterior will be evaluated for each parameter at:

**a**: Exactly at :math:`0.5`.

**b**: Sampled from the reference pdf: a Gaussian centred at :math:`0` with standard deviation :math:`0.1`.

**c**: From the prior, since there is no reference pdf: sampled uniformly in the interval :math:`[-1, 1]`.

**d**: From the ``override``, which takes precedence above all else.

.. note::

   If using this sampler **cobaya** appears to be stuck, this normally means that it cannot sample a point with finite posterior value. Check that your prior/likelihood definitions leave room for some finite posterior density, e.g. don't define an external prior that imposes that :math:`x>2` if the range allowed for :math:`x` is just :math:`[0,1]`.


Evaluate sampler class
-----------------------

.. automodule:: samplers.evaluate.evaluate
   :noindex:

.. autoclass:: samplers.evaluate.evaluate
   :members:
