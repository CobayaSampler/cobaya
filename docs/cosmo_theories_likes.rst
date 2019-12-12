Cosmological theory codes and likelihoods
=========================================

Models in Cosmology are usually split in two: :math:`\mathcal{M}=\mathcal{T}+\mathcal{E}`, where

* :math:`\mathcal{T}`, the *theoretical* model, is used to compute observable quantities :math:`\mathcal{O}`
* :math:`\mathcal{E}`, the *experimental* model, accounts for instrumental errors, foregrounds... when comparing the theoretical observable with some data :math:`\mathcal{D}`.

In practice the theoretical model is encapsulated in one or more **theory codes** (:doc:`CLASS <theory_class>`, :doc:`CAMB <theory_camb>`...) and the experimental model in a **likelihood**, which gives the probability of the data being a realization of the given observable in the context of the experiment:

.. math::

   \mathcal{L}\left[\mathcal{D}\,|\,\mathcal{M}\right] =
   \mathcal{L}\left[\mathcal{D}\,|\,\mathcal{O},\mathcal{E}\right]


Each iteration of a sampler reproduces the model using the following steps:

#. A new set of theory+experimental parameters is proposed by the sampler.
#. The theory parameters are passed to the theory codes, which compute one or more observables.
#. The experimental parameters are passed to each of the likelihoods, which in turn ask the theory code for the current value of the observables they need, and use all of that to compute a log-probability of the data.

**cobaya** wraps the most popular cosmological codes under a single interface, documented below. The codes themselves are documented in the next sections, followed by the internal likelihoods included in **cobaya**.


Cosmological theory code
------------------------

.. autoclass:: theories._cosmo.BoltzmannBase
   :members:

.. autoclass:: theories._cosmo.PowerSpectrumInterpolator
   :members:


Cosmological theory code inheritance
--------------------------------------

.. inheritance-diagram:: theorys
   :parts: 1
