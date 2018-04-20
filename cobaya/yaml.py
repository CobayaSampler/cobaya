"""
.. module:: yaml

:Synopsis: Custom YAML loader and dumper
:Author: Jesus Torrado (parts of the code comes from stackoverflow user's)

Customization of YAML's loaded and dumper:

1. Matches 1e2 as 100 (no need for dot, or sign after e),
   from http://stackoverflow.com/a/30462009
2. Wrapper to load mappings as OrderedDict (for likelihoods and params),
   from http://stackoverflow.com/a/21912744

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os
import yaml
import re
from collections import OrderedDict as odict
import numpy as np

# Local
from cobaya.conventions import _force_reproducible


# Exceptions

class InputSyntaxError(Exception):
    """Syntax error in YAML input."""


class OutputError(Exception):
    """Error when dumping YAML info."""


def yaml_load(text_stream, Loader=yaml.Loader, object_pairs_hook=odict, file_name=None):
        class OrderedLoader(Loader):
            pass
        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
        OrderedLoader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        try:
            return yaml.load(text_stream, OrderedLoader)
        # Redefining the general exception to give more user-friendly information
        except yaml.YAMLError as exception:
            errstr = "Error in your input file "+("'"+file_name+"'" if file_name else "")
            if hasattr(exception, "problem_mark"):
                line = 1 + exception.problem_mark.line
                column = 1 + exception.problem_mark.column
                signal = " --> "
                signal_right = "    <---- "
                sep = "|"
                context = 4
                lines = text_stream.split("\n")
                pre = ((("\n"+" "*len(signal)+sep).join(
                    [""]+lines[max(line-1-context,0):line-1])))+"\n"
                errorline = (signal+sep+lines[line-1] +
                             signal_right + "column %s"%column)
                post = ((("\n"+" "*len(signal)+sep).join(
                    [""]+lines[line+1-1:min(line+1+context-1,len(lines))])))+"\n"
                raise InputSyntaxError(
                    errstr + " at line %d, column %d."%(line, column) +
                    pre + errorline + post +
                    "Maybe inconsistent indentation, '=' instead of ':', "
                    "no space after ':', or a missing ':' on an empty group?")
            else:
                raise InputSyntaxError(errstr)


def yaml_load_file(file_name):
    """Wrapper to load a yaml file."""
    with open(file_name,"r") as file:
        lines = "".join(file.readlines())
    return yaml_load(lines, file_name=file_name)


def yaml_dump(data, trim_params_info=False, force_reproducible=True,
              stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    # Dump OrderedDict's as plain dictionaries, but keeping the order
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())
    OrderedDumper.add_representer(odict, _dict_representer)
    # Dump tuples as yaml "sequences"
    def _tuple_representer(dumper, data):
        return dumper.represent_sequence(
            yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, list(data))
    OrderedDumper.add_representer(tuple, _tuple_representer)
    # Numpy arrays and numbers
    def _numpy_array_representer(dumper, data):
        return dumper.represent_sequence(
            yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, data.tolist())
    OrderedDumper.add_representer(np.ndarray, _numpy_array_representer)
    def _numpy_int_representer(dumper, data):
        return dumper.represent_int(data)
    OrderedDumper.add_representer(np.int64, _numpy_int_representer)
    def _numpy_float_representer(dumper, data):
        return dumper.represent_float(data)
    OrderedDumper.add_representer(np.float64, _numpy_float_representer)
    # Dummy representer that raises an error if object has no explicit representer
    # e.g. useful to avoid dumping a python function (makes loading impossible later)
    # NB: Necessary in pyyaml = 3.12, but apparently in later versions they would produce
    #     a `yaml.representer.RepresenterError` without needing to do this.
    #     Update (and also required version in setup.py) when next pyyaml version is out
    def _fail_representer(dumper, data):
        raise OutputError(
            "You are probably trying to dump a function into YAML, "
            "and that's not possible. If you are using an external prior "
            "or likelihood, try writing it as a string instead -- there is always a way: "
            "see the documentation about user-defined priors/likelihoods. "
            "If you *really* need to generate output, "
            "set '%s: False'"%_force_reproducible +
            " in your input, but mind that the 'full' yaml file generated cannot be "
            "called again, since it will contain unknown symbols.")
    if force_reproducible:
        OrderedDumper.add_representer(type(lambda: None), _fail_representer)
        OrderedDumper.add_multi_representer(object, _fail_representer)
    # For full infos: trim known params of each likelihood: for internal use only
    if trim_params_info:
        from copy import deepcopy
        from cobaya.conventions import _likelihood, _params
        data = deepcopy(data)
        for lik_info in data.get(_likelihood, {}).values():
            if hasattr(lik_info, "pop"):
                lik_info.pop(_params, None)
    # Dump!
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def yaml_dump_file(data, file_name):
    if os.path.isfile(file_name):
        raise ValueError("File exists: '%s'"%file_name)
    with open(file_name, "w") as f:
        f.write(yaml_dump(data))
