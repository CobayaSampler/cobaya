"""
.. module:: run

:Synopsis: Main scripts to run the sampling
:Author: Jesus Torrado

"""

# Global
from collections.abc import Mapping

# Local
from cobaya import __version__
from cobaya.conventions import kinds, _prior, _params, _version
from cobaya.conventions import _path_install, _debug, _debug_file, _output_prefix
from cobaya.conventions import _resume, _timing, _debug_default, _force, _post
from cobaya.conventions import _yaml_extensions, _separator_files, _updated_suffix
from cobaya.conventions import _modules_path_arg, _modules_path_env, _resume_default
from cobaya.output import get_output
from cobaya.model import Model
from cobaya.sampler import get_sampler_class, check_sampler_info
from cobaya.log import logger_setup, LoggedError
from cobaya.yaml import yaml_dump
from cobaya.input import update_info
from cobaya.mpi import import_MPI, is_main_process, set_mpi_disabled, sync_processes
from cobaya.tools import warn_deprecation, recursive_update
from cobaya.post import post


def run(info, _from_script=False, _no_warn=False):
    # This function reproduces the model-->output-->sampler pipeline one would follow
    # when instantiating by hand, but alters the order to performs checks and dump info
    # as early as possible, e.g. to check if resuming possible or `force` needed.
    assert isinstance(info, Mapping), (
        "The first argument must be a dictionary with the info needed for the run. "
        "If you were trying to pass the name of an input file instead, "
        "load it first with 'cobaya.input.load_input', "
        "or, if you were passing a yaml string, load it with 'cobaya.yaml.yaml_load'.")
    logger_setup(info.get(_debug), info.get(_debug_file))
    import logging
    logger_run = logging.getLogger(__name__.split(".")[-1])
    # 1. Prepare output driver, if requested by defining an output_prefix
    output = get_output(output_prefix=info.get(_output_prefix),
                        resume=info.get(_resume), force=info.get(_force))
    # 2. Update the input info with the defaults for each component
    updated_info = update_info(info)
    if logging.root.getEffectiveLevel() <= logging.DEBUG:
        # Dump only if not doing output (otherwise, the user can check the .updated file)
        if not output and is_main_process():
            logger_run.info(
                "Input info updated with defaults (dumped to YAML):\n%s",
                yaml_dump(updated_info))
    # 3. If output requested, check compatibility if existing one, and dump.
    # 3.1 First: model only
    output.check_and_dump_info(info, updated_info, cache_old=True, ignore_blocks=[kinds.sampler])
    # 3.2 Then sampler -- 1st get the last sampler mentioned in the updated.yaml
    # TODO: ideally, using Minimizer would *append* to the sampler block.
    #       Some code already in place, but not possible at the moment.
    try:
        last_sampler = list(updated_info[kinds.sampler])[-1]
        last_sampler_info = {last_sampler: updated_info[kinds.sampler][last_sampler]}
    except (KeyError, TypeError):
        raise LoggedError(logger_run, "No sampler requested.")
    sampler_class = get_sampler_class(last_sampler_info)
    check_sampler_info(
        (output.reload_updated_info(use_cache=True) or {}).get(kinds.sampler),
        updated_info[kinds.sampler])
    # Dump again, now including sampler info
    output.check_and_dump_info(info, updated_info, check_compatible=False)
    # Check if resumible run
    sampler_class.check_force_resume(
        output, info=updated_info[kinds.sampler][sampler_class.__name__])
    # 4. Initialize the posterior and the sampler
    with Model(updated_info[_params], updated_info[kinds.likelihood],
               updated_info.get(_prior), updated_info.get(kinds.theory),
               path_install=info.get(_path_install), timing=updated_info.get(_timing),
               allow_renames=False, stop_at_error=info.get("stop_at_error", False)) \
               as model:
        # Re-dump the updated info, now containing parameter routes and version info
        updated_info = recursive_update(updated_info, model.info())
        output.check_and_dump_info(None, updated_info, check_compatible=False)
        with sampler_class(updated_info[kinds.sampler][sampler_class.__name__], model,
                output, path_install=info.get(_path_install)) as sampler:
            # Re-dump updated info, now also containing updates from the sampler
            updated_info[kinds.sampler][sampler.get_name()] = \
                recursive_update(
                    updated_info[kinds.sampler][sampler.get_name()], sampler.info())
            # TODO -- maybe also re-dump model info, now possibly with measured speeds
            # (waiting until the camb.transfers issue is solved)
            output.check_and_dump_info(None, updated_info, check_compatible=False)
            # Run the sampler
            sampler.run()
    # To be deprecated in the future
    # (but leave `_no_warn` for one more release with no effect and a deprecation warning)
    if not _from_script and not _no_warn and is_main_process():
        logger_run.warning(
            "The variables returned by this function have changed since the last version:"
            " they were `(updated_info, sampler.products())` and they are now "
            "`(updated_info, model, sampler)`. "
            "(To turn off this warning, pass `_no_warn=True`.)")
    return updated_info, model, sampler


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
    parser.add_argument("--no-mpi", action='store_true',
                        help="disable MPI when mpi4py installed but MPI does "
                             "not actually work")
    args = parser.parse_args()
    if args.no_mpi:
        set_mpi_disabled()
    if any((os.path.splitext(f)[0] in ("input", "updated")) for f in args.input_file):
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
            raise ValueError("Not a valid input file, or non-existent run to resume")
        # We need to update the output_prefix to resume the run *where it is*
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
        run(info, _from_script=True)
