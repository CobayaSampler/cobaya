"""
.. module:: yaml

:Synopsis: Custom YAML loader and dumper
:Author: Jesus Torrado (parts of the code comes from stackoverflow user's)

Customization of YAML's loaded and dumper:

1. Matches 1e2 as 100 (no need for dot, or sign after e),
   from https://stackoverflow.com/a/30462009
2. Wrapper to load mappings as OrderedDict (for likelihoods and params),
   from https://stackoverflow.com/a/21912744

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
from cobaya.tools import prepare_comment, recursive_update
from cobaya.conventions import _yaml_extensions


# Exceptions #############################################################################

class InputSyntaxError(Exception):
    """Syntax error in YAML input."""


class OutputError(Exception):
    """Error when dumping YAML info."""


# Custom loader ##########################################################################

class ScientificLoader(yaml.Loader):
    pass


ScientificLoader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


class OrderedLoader(ScientificLoader):
    pass


def _construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return odict(loader.construct_pairs(node))


OrderedLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


class DefaultsLoader(OrderedLoader):
    pass


def _construct_defaults(loader, node):
    if current_file_name is None:
        raise InputSyntaxError(
            "'!defaults' directive can only be used when loading from a file.")
    try:
        defaults_files = [loader.construct_scalar(node)]
    except yaml.constructor.ConstructorError:
        defaults_files = loader.construct_sequence(node)
    folder = os.path.dirname(current_file_name)
    loaded_defaults = odict()
    for dfile in defaults_files:
        dfilename = os.path.abspath(os.path.join(folder, dfile))
        try:
            dfilename += next(ext for ext in [""] + list(_yaml_extensions)
                              if (os.path.basename(dfilename) + ext
                                  in os.listdir(os.path.dirname(dfilename))))
        except StopIteration:
            raise InputSyntaxError("Mentioned non-existent defaults file '%s', "
                                   "searched for in folder '%s'." % (dfile, folder))
        this_loaded_defaults = yaml_load_file(dfilename)
        loaded_defaults = recursive_update(loaded_defaults, this_loaded_defaults)
    return loaded_defaults


DefaultsLoader.add_constructor('!defaults', _construct_defaults)


def yaml_load(text_stream, file_name=None):
    try:
        # Use a global to store the file name, to be used to locate defaults files
        # (a bit hacky, but it works)
        global current_file_name
        current_file_name = file_name
        return yaml.load(text_stream, DefaultsLoader)
    # Redefining the general exception to give more user-friendly information
    except (yaml.YAMLError, TypeError) as exception:
        errstr = "Error in your input file " + ("'" + file_name + "'" if file_name else "")
        if hasattr(exception, "problem_mark"):
            line = 1 + exception.problem_mark.line
            column = 1 + exception.problem_mark.column
            signal = " --> "
            signal_right = "    <---- "
            sep = "|"
            context = 4
            lines = text_stream.split("\n")
            pre = ((("\n" + " " * len(signal) + sep).join(
                [""] + lines[max(line - 1 - context, 0):line - 1]))) + "\n"
            errorline = (signal + sep + lines[line - 1] +
                         signal_right + "column %s" % column)
            post = ((("\n" + " " * len(signal) + sep).join(
                [""] + lines[line + 1 - 1:min(line + 1 + context - 1, len(lines))]))) + "\n"
            raise InputSyntaxError(
                errstr + " at line %d, column %d." % (line, column) +
                pre + errorline + post +
                "Maybe inconsistent indentation, '=' instead of ':', "
                "no space after ':', or a missing ':' on an empty group?")
        else:
            raise InputSyntaxError(errstr)


def yaml_load_file(file_name):
    """Wrapper to load a yaml file.

    Manages !defaults directive."""
    with open(file_name, "r") as file:
        lines = "".join(file.readlines())
    return yaml_load(lines, file_name=file_name)


# Custom dumper ##########################################################################

def yaml_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
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

    # Dummy representer that prints True for non-representable python objects
    # (prints True instead of nothing because some functions try cast values to bool)
    def _null_representer(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:bool', 'true')

    OrderedDumper.add_representer(type(lambda: None), _null_representer)
    OrderedDumper.add_multi_representer(object, _null_representer)

    # Dump!
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def yaml_dump_file(file_name, data, comment=None, error_if_exists=True):
    if error_if_exists and os.path.isfile(file_name):
        raise IOError("File exists: '%s'" % file_name)
    with open(file_name, "w+") as f:
        if comment:
            f.write(prepare_comment(comment))
        f.write(yaml_dump(data))
