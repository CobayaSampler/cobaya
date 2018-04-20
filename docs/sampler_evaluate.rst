Samplers – evaluate
===================

This is a *dummy* sampler that just evaluates the likelihood at a *reference* point. You can use it to test your likelihoods.

To use it, simply make the ``sampler`` block:

.. code-block:: yaml

   sampler:
     evaluate:

If you want to fix the evaluation point, give just a value as a *reference*
for the parameters in the input file. Otherwise, it will be sampled from the *reference* pdf or, if it does not exists, from the *prior*: For example:

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

In this case, the posterior will be evaluated at ``a=0.5``, ``b`` sampled from a normal
pdf centred at ``0`` with standard deviation ``0.1``, and ``c`` will be sampled uniformly
between ``-1`` and ``1``.


Sampler class
-------------

.. automodule:: samplers.evaluate.evaluate
   :noindex:

.. autoclass:: samplers.evaluate.evaluate
   :members:
