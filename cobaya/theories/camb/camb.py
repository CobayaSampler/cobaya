"""
.. module:: theories.camb

:Synopsis: Managing the CAMB cosmological code
:Author: Jesus Torrado

.. |br| raw:: html

   <br />

This module imports and manages the CAMB cosmological code.

.. note::

   **If you use this cosmological code, please cite it as:**
   |br|
   `A. Lewis, A. Challinor, A. Lasenby, "Efficient computation of CMB anisotropies in closed FRW"
   (arXiv:astro-ph/9911177) <https://arxiv.org/abs/astro-ph/9911177>`_
   |br|
   `C. Howlett, A. Lewis, A. Hall, A. Challinor, "CMB power spectrum parameter degeneracies in the era of precision cosmology"
   (arXiv:1201.3654) <https://arxiv.org/abs/1201.3654>`_


Usage
-----

If you are using a likelihood that requires some observable from CAMB, simply add CAMB
to the theory block.

You can specify any parameter that CAMB understands within the ``theory``
sub-block of the ``params`` block:

.. code-block:: yaml

   theory:
     camb:

   params:
     theory:
       [any param that CAMB understands, fixed, sampled or derived]


Installation
------------

Pre-requisites
^^^^^^^^^^^^^^

**cobaya** calls CAMB using its Python interface, which requires that you compile CAMB
using the GNU gfortran compiler version 4.9 or later. To check if you fulfil that
requisite, type ``gfortran --version`` in the shell, and the first line should look like

.. code::

   GNU Fortran ([your OS version]) [gfortran version] [release date]

Check that ``[gfortran's version]`` is at least 4.9. If you get an error instead, you need
to install gfortran (contact your local IT service).


Automatic installation
^^^^^^^^^^^^^^^^^^^^^^

If you do not plan to modify CAMB, the easiest way to install it is using the
:doc:`automatic installation script <installation_ext>`. Just make sure that
``theory: camb:`` appears in one of the files passed as arguments to the installation
script.


Manual installation (or using your own version)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are planning to modify CAMB or use an already modified version,
you should not use the automatic installation script. Use the installation method that
best adapts to your needs:

* [**Recommended**]
  To install an up-to-date CAMB locally and use git to keep track of your changes,
  `fork the CAMB repository in Github <https://github.com/cmbant/CAMB>`_
  (follow `this instructions <https://help.github.com/articles/fork-a-repo/>`_) and clone
  it in some folder of your choice, say ``/path/to/theories/CAMB``:

  .. code:: bash
     
      $ cd /path/to/theories/
      $ git clone https://[YourGithubUser]@github.com/[YourGithubUser]/CAMB.git
      $ cd CAMB/pycamb
      $ python setup.py build
     
* To install an up-to-date CAMB locally, if you don't care about keeping track of your
  changes (and don't want to use ``git``), do:

  .. code:: bash

      $ cd /path/to/theories/
      $ wget https://github.com/cmbant/CAMB/archive/master.zip
      $ unzip master.zip
      $ rm master.zip
      $ mv CAMB-master CAMB
      $ cd CAMB/pycamb
      $ python setup.py build

* To use your own version, assuming it's placed under ``/path/to/theories/CAMB``,
  just make sure it is compiled (and that the version on top of which you based your
  modifications is old enough to have the ``pycamb`` interface implemented.

In the three cases above, you **must** specify the path to your CAMB installation in
the input block for CAMB (otherwise a system-wide CAMB may be used instead):

.. code:: yaml

   theory:
     camb:
       path: /path/to/theories/CAMB

.. note::

   In any of these methods, you should **not** install CAMB as python package using
   ``python setup.py install --user``, as the official instructions suggest.
   It is actually safer not to do so if you intend to switch between different versions or
   modifications of CAMB.


Modifying CAMB
--------------

If you modify CAMB and add new variables, you don't need to let **cobaya** now,
but make sure
that the variables you create are exposed in its Python interface (contact CAMB's
developers if you need help with that).

.. todo::

   Point somewhere to the CAMB documentation where how to make these modifications
   is explained.

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import sys
import os
from copy import deepcopy
import numpy as np
from collections import namedtuple, OrderedDict as odict

# Local
from cobaya.theory import Theory
from cobaya.log import HandledException
from cobaya.install import user_flag_if_needed

# Logger
import logging
log = logging.getLogger(__name__)

# Result collector
collector = namedtuple("collector", ["method", "kwargs"])


class camb(Theory):

    def initialise(self):
        """Importing CAMB from the correct path, if given."""
        if self.path:
            log.info("Importing *local* CAMB from "+self.path)
            if not os.path.exists(self.path):
                log.error("The given folder does not exist: '%s'", self.path)
                raise HandledException
            pycamb_path = os.path.join(self.path, "pycamb")
            if not os.path.exists(pycamb_path):
                log.error("Either CAMB is not in the given folder, '%s', or you are using"
                          " a very old version without the `pycamb` Python interface.",
                          self.path)
                raise HandledException
            sys.path.insert(0, pycamb_path)
        else:
            log.info("Importing *global* CAMB.")
        try:
            import camb
        except ImportError:
            log.error(
                "Couldn't find the CAMB python interface.\n"
                "Make sure that you have compiled it, and that you either\n"
                " (a) specify a path (you didn't) or\n"
                " (b) install the Python interface globally with\n"
                "     '/path/to/camb/pycamb/python setup.py install --user'")
            raise HandledException
        self.camb = camb
        # Prepare errors
        from camb.baseconfig import CAMBParamRangeError
        global CAMBParamRangeError
        # Generate states, to avoid recomputing
        self.n_states = 3
        self.states = [{"params": None, "derived": None, "derived_extra": None,
                        "last": 0} for i in range(self.n_states)]
        # Dict of named tuples to collect requirements and computation methods
        self.collectors = {}
        # Additional input parameters to pass to CAMB
        self.also = {}
        # patch: if cosmomc_theta is used, CAMB needs to be passed explicitly "H0=None"
        if all((p in self.input_params) for p in ["H0", "cosmomc_theta"]):
            log.error("Can't pass both H0 and cosmomc_theta to Camb.")
            raise HandledException
        if "cosmomc_theta" in self.input_params or "cosmomc_theta":
            self.also["H0"] = None
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []

    def current_state(self):
        lasts = [self.states[i]["last"] for i in range(self.n_states)]
        return self.states[lasts.index(max(lasts))]

    def needs(self, arguments):
        # Computed quantities required by the likelihood
        for k,v in arguments.items():
            # Precision parameters and boundaries (in general, take max of all requested)
            if k == "l_max":
                self.also["lmax"] = max(v, self.also.get("lmax",0))
            elif k == "k_max":
                self.also["kmax"] = max(v, self.also.get("kmax",0))
            elif k == "redshifts":
                self.also["redshifts"] = np.sort(
                    np.unique(np.concatenate((v, self.also.get("redshifts",[])))))
            # Products and other computations
            elif k == "Cl":
                self.collectors["Cl"] = collector(
                    method="CAMBdata.get_cmb_power_spectra",
                    kwargs={"spectra": ["total"], "raw_cl": True})
                self.derived_extra += ["TCMB"]
            elif k == "Pk_interpolator":
                vars_pairs = v.pop("vars_pairs", None)
                vars_pairs = vars_pairs or [["total", "total"]]
                for pair in vars_pairs:
                    name = "Pk_interpolator_%s_%s"%(pair[0],pair[1])
                    kwargs = deepcopy(v)
                    kwargs.update(dict(zip(["var1", "var2"], pair)))
                    self.collectors[name] = collector(
                        method="CAMBdata.get_matter_power_interpolator",
                        kwargs=kwargs)
            else:
                # Extra derived parameters
                if v is None:
                    self.derived_extra += [k]
                else:
                    log.error("'%s' does not understand the requirement '%s:%s'.",
                              self.__class__.__name__,k,v)
                    raise HandledException
        # Derived parameters (if some need some additional computations)
        # ...

    def set(self, params_values_dict, i_state):
        # Store them, to use them later to identify the state
        self.states[i_state]["params"] = params_values_dict
        # Feed the arguments defining the cosmology to the cosmological code
        # Additionally required input parameters
        args = deepcopy(self.also)
        # Sampled -- save the state for avoiding recomputing later
        args.update(params_values_dict)
        # Precision (fixed at the theory block level)
        if self.precision:
            args.update(self.precision)
        # Generate and save
        log.debug("Setting parameters: %r", args)
        try:
            return self.camb.set_params(**args)
        except CAMBParamRangeError:
            print("CAMBParamRangeError!!! It should fail here!")
            return None
        except:
            log.error("Error setting CAMB parameters -- see CAMB's error trace below.\n"
                      "The parameters were %r", args)
            raise  # No HandledException, so that CAMB traceback gets printed

    def compute(self, derived=None, **params_values_dict):
        lasts = [self.states[i]["last"] for i in range(self.n_states)]
        try:
            # are the parameter values there already?
            i_state = (i for i in range(self.n_states)
                       if self.states[i]["params"] == params_values_dict).next()
            # Get (pre-computed) derived parameters
            if derived == {}:
                derived.update(self.states[i_state]["derived"])
            log.debug("Re-using computed results (state %d)", i_state)
        except StopIteration:
            # update the (first) oldest one and compute
            i_state = lasts.index(min(lasts))
            log.debug("Computing (state %d)", i_state)
            intermediates = {
                "CAMBparams": {"result": self.set(params_values_dict, i_state)},
                "CAMBdata": {"method": "get_results", "result": None}}
            # Compute the necessary products
            for product, collector in self.collectors.items():
                parent, method = collector.method.split(".")
                if intermediates[parent]["result"] is None:
                    intermediates[parent]["result"] = getattr(
                        self.camb, intermediates[parent]["method"])(
                            intermediates["CAMBparams"]["result"])
                method = getattr(intermediates[parent]["result"], method)
                self.states[i_state][product] = method(**self.collectors[product].kwargs)
            # Prepare derived parameters
            if derived == {}:
                derived.update(self.get_derived_all(intermediates))
                self.states[i_state]["derived"] = odict(
                    [[p,derived[p]] for p in self.output_params])
            # Prepare necessary extra derived parameters
            self.states[i_state]["derived_extra"] = {
                p:self.get_derived(p, intermediates) for p in self.derived_extra}
        # make this one the current one by decreasing the antiquity of the rest
        for i in range(self.n_states):
            self.states[i]["last"] -= max(lasts)
        self.states[i_state]["last"] = 1
        return True

    def get_derived_from_params(self, p, intermediates):
        return getattr(intermediates["CAMBparams"]["result"], p, None)

    def get_derived_from_std(self, p, intermediates):
        return intermediates["CAMBdata"]["result"].get_derived_params().get(p, None)

    def get_derived_from_getter(self, p, intermediates):
        return getattr(intermediates["CAMBparams"]["result"], "get_"+p, lambda: None)()

    def get_derived(self, p, intermediates):
        """
        General function to extract a single derived parameter.

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        # Specific calls, if general ones fail:
        if p == "sigma8":
            return intermediates["CAMBdata"]["result"].get_sigma8()[0]
        for f in [self.get_derived_from_params,
                  self.get_derived_from_std,
                  self.get_derived_from_getter]:
            derived = f(p, intermediates)
            if derived is not None:
                return derived

    def get_derived_all(self, intermediates):
        """
        Returns a dictionary of derived parameters with their values,
        using the *current* state.

        To get a parameter *from a likelihood* use `get_param` instead.
        """
        derived = {}
        for p in self.output_params:
            derived[p] = self.get_derived(p, intermediates)
            if derived[p] is None:
                log.error("Derived param '%s' not implemented in the CAMB interface", p)
                raise HandledException
        return derived

    def get_param(self, p):
        """
        Interface function for likelihoods to get sampled and derived parameters.
        """
        current_state = self.current_state()
        for pool in ["params", "derived", "derived_extra"]:
            value = current_state[pool].get(p, None)
            if value is not None:
                return value
        log.error("Parameter not known: '%s'", p)
        raise HandledException

    def get_cl(self, ell_factor=False):
        """
        Returns the power spectra in microK^2,
        using the *current* state.
        """
        current_state = self.current_state()
        # get C_l^XX from the cosmological code
        cl_camb = deepcopy(current_state["Cl"]["total"])
        mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3}
        cl = {"ell": np.arange(cl_camb.shape[0])}
        cl.update({sp: cl_camb[:,i] for sp,i in mapping.items()})
        # Lensing -- TODO!
        # pp <-- camb_results.get_lens_potential_cls(lmax)[:, 0]
        ell_factor = ((cl["ell"]+1)*cl["ell"]/(2*np.pi))[2:] if ell_factor else 1
        # convert dimensionless C_l's to C_l in muK**2
        T = current_state["derived_extra"]["TCMB"]
        for key in cl.keys():
            if key not in ['pp', 'ell']:
                cl[key][2:] = cl[key][2:] * (T*1.e6)**2 * ell_factor
        return cl

    def get_Pk_interpolator(self):
        current_state = self.current_state()
        prefix = "Pk_interpolator_"
        return {k[len(prefix):]:v for k,v in current_state.items() if k.startswith(prefix)}


# Installation routines ##################################################################

def is_installed(**kwargs):
    try:
        if kwargs["code"]:
            import camb
    except:
        return False
    return True


def install(force=False, code=True, **kwargs):
    if not code:
        log.info("Code not requested. Nothing to do.")
        return True
    import pip
    exit_status = pip.main(
        ["install", "camb", "--upgrade"] + user_flag_if_needed() +
        (["--force-reinstall"] if force else []))
    # Notice that --force-reinstall only works when --upgrade is present too!
    if exit_status:
        return False
    return True
