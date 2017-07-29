Samplers -- MCMC
================

.. automodule:: samplers.mcmc.mcmc
   :noindex:

Options and defaults
--------------------

Simply copy this block in your input ``yaml`` file and modify whatever options you want (you can delete the rest).

.. literalinclude:: ../cobaya/samplers/mcmc/defaults.yaml
   :language: yaml

           
Sampler class
-------------
   
.. autoclass:: samplers.mcmc.mcmc
   :members:

Proposal
--------      

.. automodule:: samplers.mcmc.proposal
   :noindex:
   
.. autoclass:: samplers.mcmc.proposal.CyclicIndexRandomizer                     
   :members:
.. autoclass:: samplers.mcmc.proposal.RandDirectionProposer
   :members:
.. autoclass:: samplers.mcmc.proposal.BlockedProposer                     
   :members:

