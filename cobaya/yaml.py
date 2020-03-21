"""
.. module:: yaml

:Synopsis: Custom YAML loader and dumper
:Author: Jesus Torrado (parts of the code comes from stackoverflow user's)

Customization of YAML's loaded and dumper:

1. Matches 1e2 as 100 (no need for dot, or sign after e),
   from https://stackoverflow.com/a/30462009

"""
# Global
import os
import re
import yaml
import numpy as np
from yaml.resolver import BaseResolver
from yaml.constructor import ConstructorError
from collections import OrderedDict
from typing import Mapping

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


class DefaultsLoader(ScientificLoader):
    current_folder = None


def _construct_defaults(loader, node):
    if loader.current_folder is None:
        raise InputSyntaxError(
            "'!defaults' directive can only be used when loading from a file.")
    try:
        defaults_files = [loader.construct_scalar(node)]
    except ConstructorError:
        defaults_files = loader.construct_sequence(node)
    folder = loader.current_folder
    loaded_defaults = {}
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
        # set current_folder to store the file name, to be used to locate relative
        # defaults files
        DefaultsLoader.current_folder = os.path.dirname(file_name) if file_name else None
        return yaml.load(text_stream, DefaultsLoader)
    # Redefining the general exception to give more user-friendly information
    except (yaml.YAMLError, TypeError) as exception:
        errstr = "Error in your input file " + (
            "'" + file_name + "'" if file_name else "")
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
                [""] + lines[
                       line + 1 - 1:min(line + 1 + context - 1, len(lines))]))) + "\n"
            bullet = "\n- "
            raise InputSyntaxError(
                errstr + " at line %d, column %d." % (line, column) +
                pre + errorline + post +
                "Some possible causes:" + bullet +
                bullet.join([
                    "inconsistent indentation", "'=' instead of ':'",
                    "no space after ':'", "a missing ':'", "an empty group",
                    "'\' in a double-quoted string (\") not starting by 'r\"'."]))
        else:
            raise InputSyntaxError(errstr)


def yaml_load_file(file_name, yaml_text=None):
    """Wrapper to load a yaml file.

    Manages !defaults directive."""
    if yaml_text is None:
        with open(file_name, "r", encoding="utf-8-sig") as file:
            yaml_text = "".join(file.readlines())
    return yaml_load(yaml_text, file_name=file_name)


# Custom dumper ##########################################################################

def yaml_dump(info, stream=None, Dumper=yaml.Dumper, **kwds):
    class CustomDumper(Dumper):
        pass

    # Make sure dicts preserve order when dumped
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    CustomDumper.add_representer(dict, _dict_representer)
    CustomDumper.add_representer(Mapping, _dict_representer)
    CustomDumper.add_representer(OrderedDict, _dict_representer)

    # Dump tuples as yaml "sequences"
    def _tuple_representer(dumper, data):
        return dumper.represent_sequence(
            BaseResolver.DEFAULT_SEQUENCE_TAG, list(data))

    CustomDumper.add_representer(tuple, _tuple_representer)

    # Numpy arrays and numbers
    def _numpy_array_representer(dumper, data):
        return dumper.represent_sequence(
            BaseResolver.DEFAULT_SEQUENCE_TAG, data.tolist())

    CustomDumper.add_representer(np.ndarray, _numpy_array_representer)

    def _numpy_int_representer(dumper, data):
        return dumper.represent_int(data)

    CustomDumper.add_representer(np.int64, _numpy_int_representer)

    def _numpy_float_representer(dumper, data):
        return dumper.represent_float(data)

    CustomDumper.add_representer(np.float64, _numpy_float_representer)

    # Dummy representer that prints True for non-representable python objects
    # (prints True instead of nothing because some functions try cast values to bool)
    def _null_representer(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:bool', 'true')

    CustomDumper.add_representer(type(lambda: None), _null_representer)
    CustomDumper.add_multi_representer(object, _null_representer)

    # Dump!
    return yaml.dump(info, stream, CustomDumper, allow_unicode=True, **kwds)


def yaml_dump_file(file_name, data, comment=None, error_if_exists=True):
    if error_if_exists and os.path.isfile(file_name):
        raise IOError("File exists: '%s'" % file_name)
    with open(file_name, "w+", encoding="utf-8") as f:
        if comment:
            f.write(prepare_comment(comment))
        f.write(yaml_dump(data))
