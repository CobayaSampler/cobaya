Likelihoods – One
=================

Likelihoods that evaluates to 1. Useful to explore priors and to compute prior volumes – see :ref:`polychord_bayes_ratios`.

Usage
-----

Simply copy this block in your input ``yaml`` file and modify whatever options you want
(you can delete the rest).

.. literalinclude:: ../cobaya/likelihoods/one/one.yaml
   :language: yaml


The option ``prefix`` fixes the parameters that will be understood by this likelihood (see :ref:`likehood_mock_params`).              


``one`` likelihood class
------------------------

.. automodule:: likelihoods.one.one
   :noindex:

.. autoclass:: likelihoods.one.one
   :members:
