"""
if I want to sample from f(x) but the lik knows x, I need to 3 things:
1. x is the parameter known by the likelihood
2. x = exp(logx)
3. I must drop logx before passing the parameters to the likelihood.

New proposed syntax (update development notes!)

params:
  logx:
    prior: [...]
    drop:
  x: lambda logx: np.exp(x)

Now, loading params info consists of creating:
- A list of "input" parameters, which is exactly the same as the "sampled" ones:
  --> those that have a "prior"
- A list of "likelihood" parameters: those that will be passed tp the likelihood;
  Identified because they are not derived, i.e., are assigned a value (not a dict)
  or a dict that does not contain a "prior".
- A list of functions to translate between both.

OJO: pasar los fijos al inicializar, porque puede que cambie la forma de inicializar segun
el valor de los fijos!


CONFLICTO:

No esta claso, si escribo
a: f(c)
b: g(c)
cual es IN y cual es OUT:

Dos soluciones: separar bloques in y out en INPUT, o forzar OUT (derived) to be
a: f(c)
b:
   derived: g(c)


        # Assume that the *un*known function arguments are likelihood output parameters

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from six import string_types

# Global
from collections import OrderedDict as odict
from numbers import Number
from inspect import getargspec
from itertools import chain

# Local
from cobaya.conventions import _prior, _p_drop, _p_derived, _theory
from cobaya.tools import get_external_function

def is_fixed_param(info_param):
    """
    Returns True if `info_param` is a number, a string or a function.
    """
    return (isinstance(info_param, Number) or isinstance(info_param, string_types)
            or callable(info_param))

def is_sampled_param(info_param):
    """
    Returns True if `info_param` has a `%s` field.
    """%_prior
    return _prior in (info_param if hasattr(info_param, "get") else {})

def is_derived_param(info_param):
    """
    Returns False if `info_param` is "fixed" or "sampled".
    """
    return not(is_fixed_param(info_param) or is_sampled_param(info_param))

class Parametrisation(object):
    """
    Class managing parametrisation. Translates parameter between sampler+prior and likelihood
    """
    def __init__(self, info_params):
        # First, we load the parameters, not caring about whether they are understood by any likelihood
        # `input` contains the parameters (expected to be) understood by the likelihood,
        #   with its fixed value, its fixing function, or None if their value is given
        #   directly by the sampler.
        self._input = odict()
        self._input_funcs = dict()
        self._input_args = dict()
        self._output = odict()
        self._sampled = odict()
        self._derived = odict()
        self._derived_funcs = dict()
        self._derived_args = dict()
        self._theory_params = info_params.get(_theory,{}).keys()
        info_params_flat = odict([(p,info_params[_theory][p]) for p in self._theory_params])
        info_params_flat.update(odict([(p,info_params[p]) for p in info_params if p!=_theory]))
        for p, info in info_params_flat.iteritems():
            if is_fixed_param(info):
                self._input[p] = info if isinstance(info, Number) else None
                if not self._input[p]:
                    self._input_funcs[p] = get_external_function(info)
                    self._input_args[p] = getargspec(self._input_funcs[p]).args
            if is_sampled_param(info):
                self._sampled[p] = info
                if not _p_drop in info:
                    self._input[p] = None
            if is_derived_param(info):
                self._derived[p] = info
                if _p_derived in (info or {}):
                    self._derived_funcs[p] = get_external_function(info[_p_derived])
                    self._derived_args[p] = getargspec(self._derived_funcs[p]).args
                else:
                    self._output[p] = None
        # Assume that the *un*known function arguments are likelihood output parameters
        args = (set(chain(*self._input_args.values()))
                .union(chain(*self._derived_args.values())))
        for p in self._input.keys() + self._sampled.keys() + self._output.keys():
            if p in args:
                args.remove(p)
        self._output.update({p:None for p in args})
        # Useful sets: directly-sampled input parameters and directly "output-ed" derived
        self._directly_sampled = [p for p in self._input if p in self._sampled]
        self._directly_output = [p for p in self._derived if p in self._output]

    def input_params(self):
        return self._input

    def output_params(self):
        return self._output

    def sampled_params(self):
        return self._sampled

    def derived_params(self):
        return self._derived

    def theory_params(self):
        return self._theory_params
    
    def to_input(self, sampled_params_values):
        as_odict = odict(zip(self.sampled_params(), sampled_params_values))
        # Fill first directly sampled input parameters
        self.input_params().update({p:as_odict[p] for p in self._directly_sampled})
        # Then evaluate the functions
        for p in self._input_funcs:
            args = {p:self.input_params().get(p, as_odict.get(p, None))
                    for p in self._input_args[p]}
            self.input_params()[p] = self._input_funcs[p](**args)
        return self.input_params()

    def to_derived(self, output_params_values):
        as_odict = odict(zip(self.output_params(), output_params_values))
        # Fill first derived parameters which are direct output parameters
        self.derived_params().update({p:as_odict[p] for p in self._directly_output})
        # Then evaluate the functions
        for p in self._derived_funcs:
            args = {p:self.input_params().get(p, as_odict.get(p, None))
                    for p in self._derived_args[p]}
            self.derived_params()[p] = self._derived_funcs[p](**args)
        return self.derived_params().values()

    def labels(self):
        print "REMOVE CORRESPONDING FUNCTION OF 'prior'."

        # PASS NEXT 2 LINES A PARAMETRSATOR
        # Labels and reference point
#        self.labels = get_labels(sampled_params_info)
       
       

            # NEXT LINES SHOULD GO TO PARAMETRISATOR
            ### # Store the rest of the properties
            ### self.properties[p] = dict(
            ###     [(k,v) for k,v in sampled_params_info[p].iteritems()
            ###      if k not in [_prior,_p_ref,_p_label]])
            # Store prior boundaries

    # def property(self, param, prop, default=None):
    #     """
    #     Returns:
    #        The value of the field ``prop`` of in the info of parameter ``param``, it that
    #        field has been defined (otherwise, the value of ``default``).
    #     """
    #     raise NotImplementedError("DON'T CALL ME: call method from Parametrisation instead!")
    #     return self.properties[param].get(prop, default)


        
    # Python magic for the "with" statement
    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        return
