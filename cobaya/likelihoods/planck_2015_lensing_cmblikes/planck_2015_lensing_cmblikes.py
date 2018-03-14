"""
.. module:: bicep_keck_2015

:Synopsis: Likelihood of BICEP2/Keck-Array, October 2015
:Author: Antony Lewis

.. |br| raw:: html

   <br />

.. note::

   **If you use this likelihood, please cite it as:**
   |br|
   P.A.R. Ade et al,
   `Improved Constraints on Cosmology and Foregrounds from
   BICEP2 and Keck Array Cosmic Microwave Background Data
   with Inclusion of 95 GHz Band`
   `(arXiv:1510.09217) <https://arxiv.org/abs/1510.09217>`_

Usage
-----

To use this likelihood, ``bicep_keck_2015``, you simply need to mention it in the
``likelihood`` block. The corresponding nuisance parameters will be added automatically,
so you don't have to care about listing them in the ``params`` block.

An example of usage can be found in :doc:`examples_bkp`.

The nuisance parameters and their default priors can be found in the ``defaults.yaml``
files in the folder for the source code of this module, and it's reproduced below.

You shouldn't need to modify any of the options of this simple likelihood,
but if you really need to, just copy the ``likelihood`` block into your input ``yaml``
file and modify whatever options you want (you can delete the rest).

.. literalinclude:: ../cobaya/likelihoods/bicep_keck_2015/defaults.yaml
   :language: yaml


Installation
------------

This likelihood can be installed automatically as explained in :doc:`installation_cosmo`.
If are following the instructions there (you should!), you don't need to read the rest
of this section.

Manual installation of the BICEP2/Keck-Array likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming you are installing all your
likelihoods under ``/path/to/likelihoods``, simply do

.. code:: bash

   $ cd /path/to/likelihoods
   $ mkdir bicep_keck_2015
   $ cd bicep_keck_2015
   $ wget http://bicepkeck.org/BK14_datarelease/BK14_cosmomc.tgz
   $ tar xvf BK14_cosmomc.tgz
   $ rm BK14_cosmomc.tgz

After this, mention the path to this likelihood when you include it in an input file as

.. code-block:: yaml

   likelihood:
     bicep_keck_2015
       path: /path/to/likelihoods/bicep_keck_2015

"""

# Global
from __future__ import division, print_function
import os

# Local
from cobaya.likelihoods._cmblikes_prototype import _cmblikes_prototype

# Logger
import logging


class planck_2015_lensing_cmblikes(_cmblikes_prototype):

    pass


