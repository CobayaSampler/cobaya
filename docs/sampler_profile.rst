``profile`` sampler
====================

.. automodule:: samplers.profile.profile
   :noindex:

Options and defaults
--------------------

Simply copy this block in your input ``yaml`` file and modify whatever options you want (you can delete the rest).

.. literalinclude:: ../cobaya/samplers/profile/profile.yaml
   :language: yaml

Note that you must specify a parameter to profile, which must also appear in the ``params`` block. Also, you must provide either a list of values for the parameter, or the extremes of a range with the number of points within that.

Profile class
--------------
   
.. autoclass:: samplers.profile.Profile
   :members:
