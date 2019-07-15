"""
.. module:: run

:Synopsis: Main scripts to run the sampling
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
from collections import OrderedDict as odict

# Local
from cobaya import __version__
from cobaya.conventions import _likelihood, _prior, _params, _theory, _sampler
from cobaya.conventions import _path_install, _debug, _debug_file, _output_prefix
from cobaya.conventions import _resume, _timing, _debug_default, _force, _post
from cobaya.conventions import _yaml_extensions, _separator_files, _updated_suffix
from cobaya.conventions import _modules_path_arg, _modules_path_env, _resume_default
from cobaya.output import get_Output as Output
from cobaya.model import Model
from cobaya.sampler import get_sampler as Sampler
from cobaya.log import logger_setup
from cobaya.yaml import yaml_dump
from cobaya.input import update_info
from cobaya.mpi import import_MPI, am_single_or_primary_process
from cobaya.tools import warn_deprecation, deepcopy_where_possible
from cobaya.post import post


def run(info):
    assert hasattr(info, "keys"), (
        "The first argument must be a dictionary with the info needed for the run. "
        "If you were trying to pass the name of an input file instead, "
        "load it first with 'cobaya.input.load_input', "
        "or, if you were passing a yaml string, load it with 'cobaya.yaml.yaml_load'.")
    # Configure the logger ASAP
    # Just a dummy import before configuring the logger, until I fix root/individual level
    import getdist
    logger_setup(info.get(_debug), info.get(_debug_file))
    import logging
    # Initialize output, if required
    resume, force = info.get(_resume), info.get(_force)
    ignore_blocks = []
    # If minimizer, always try to re-use sample to get bestfit/covmat
    if list(info[_sampler])[0] == "minimize":
        resume = True
        force = False
    output = Output(output_prefix=info.get(_output_prefix),
                    resume=resume, force_output=force)
    # Create the updated input information, including defaults for each module.
    updated_info = update_info(info)
    if output:
        updated_info[_output_prefix] = output.updated_output_prefix()
        updated_info[_resume] = output.is_resuming()
    if logging.root.getEffectiveLevel() <= logging.DEBUG:
        # Don't dump unless we are doing output, just in case something not serializable
        # May be fixed in the future if we find a way to serialize external functions
        if info.get(_output_prefix) and am_single_or_primary_process():
            logging.getLogger(__name__.split(".")[-1]).info(
                "Input info updated with defaults (dumped to YAML):\n%s",
                yaml_dump(updated_info))
    # TO BE DEPRECATED IN >1.2!!! #####################
    _force_reproducible = "force_reproducible"
    if _force_reproducible in info:
        info.pop(_force_reproducible)
        logging.getLogger(__name__.split(".")[-1]).warning(
            "Option '%s' is no longer necessary. Please remove it!" % _force_reproducible)
    # CHECK THAT THIS WARNING WORKS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ###################################################
    # We dump the info now, before modules initialization, to better track errors and
    # to check if resuming is possible asap (old and new infos are consistent)
    output.dump_info(info, updated_info)
    # Initialize the posterior and the sampler
    with Model(updated_info[_params], updated_info[_likelihood], updated_info.get(_prior),
               updated_info.get(_theory), modules=info.get(_path_install),
               timing=updated_info.get(_timing), allow_renames=False) as model:
        # Update the updated info with the parameter routes
        keys = ([_likelihood, _theory] if _theory in updated_info else [_likelihood])
        updated_info.update(odict([[k, model.info()[k]] for k in keys]))
        output.dump_info(None, updated_info, check_compatible=False)
        with Sampler(
                updated_info[_sampler], model, output, resume=updated_info.get(_resume),
                modules=info.get(_path_install)) as sampler:
            sampler.run()
    # For scripted calls:
    # Restore the original output_prefix: the script has not changed folder!
    if _output_prefix in info:
        updated_info[_output_prefix] = info.get(_output_prefix)
    return updated_info, sampler.products()


# Command-line script
def run_script():
    warn_deprecation()
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Cobaya's run script.")
    parser.add_argument("input_file", nargs=1, action="store", metavar="input_file.yaml",
                        help="An input file to run.")
    parser.add_argument("-" + _modules_path_arg[0], "--" + _modules_path_arg,
                        action="store", nargs=1, metavar="/some/path", default=[None],
                        help="Path where modules were automatically installed.")
    parser.add_argument("-" + _output_prefix[0], "--" + _output_prefix,
                        action="store", nargs=1, metavar="/some/path", default=[None],
                        help="Path and prefix for the text output.")
    parser.add_argument("-" + _debug[0], "--" + _debug, action="store_true",
                        help="Produce verbose debug output.")
    continuation = parser.add_mutually_exclusive_group(required=False)
    continuation.add_argument("-" + _resume[0], "--" + _resume, action="store_true",
                              help="Resume an existing chain if it has similar info "
                                   "(fails otherwise).")
    continuation.add_argument("-" + _force[0], "--" + _force, action="store_true",
                              help="Overwrites previous output, if it exists "
                                   "(use with care!)")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()
    if any([(os.path.splitext(f)[0] in ("input", "updated")) for f in args.input_file]):
        raise ValueError("'input' and 'updated' are reserved file names. "
                         "Please, use a different one.")
    load_input = import_MPI(".input", "load_input")
    given_input = args.input_file[0]
    if any(given_input.lower().endswith(ext) for ext in _yaml_extensions):
        info = load_input(given_input)
        output_prefix_cmd = getattr(args, _output_prefix)[0]
        output_prefix_input = info.get(_output_prefix)
        info[_output_prefix] = output_prefix_cmd or output_prefix_input
    else:
        # Passed an existing output_prefix? Try to find the corresponding *.updated.yaml
        updated_file = (given_input +
                        (_separator_files if not given_input.endswith(os.sep) else "") +
                        _updated_suffix + _yaml_extensions[0])
        try:
            info = load_input(updated_file)
        except IOError:
            raise ValueError("Not a valid input file, or non-existent sample to resume")
        # We need to update the output_prefix to resume the sample *where it is*
        info[_output_prefix] = given_input
        # If input given this way, we obviously want to resume!
        info[_resume] = True
    # solve modules installation path cmd > env > input
    path_cmd = getattr(args, _modules_path_arg)[0]
    path_env = os.environ.get(_modules_path_env, None)
    path_input = info.get(_path_install)
    info[_path_install] = path_cmd or (path_env or path_input)
    info[_debug] = getattr(args, _debug) or info.get(_debug, _debug_default)
    info[_resume] = getattr(args, _resume, _resume_default)
    info[_force] = getattr(args, _force, False)
    if _post in info:
        post(info)
    else:
        run(info)
