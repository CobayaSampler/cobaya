"""
.. module:: run

:Synopsis: Main scripts to run the sampling
:Author: Jesus Torrado

"""

# Global
from typing import Mapping
import logging
import os

# Local
from cobaya import __version__
from cobaya.conventions import kinds, _prior, _params, _packages_path, _output_prefix, \
    _debug, _debug_file, _resume, _timing, _debug_default, _force, _post, _test_run, \
    _yaml_extensions, _packages_path_arg, \
    _packages_path_arg_posix
from cobaya.output import get_output, split_prefix, get_info_path
from cobaya.model import Model
from cobaya.sampler import get_sampler_name_and_class, check_sampler_info
from cobaya.log import logger_setup, LoggedError
from cobaya.yaml import yaml_dump, yaml_load
from cobaya.input import update_info
from cobaya.mpi import import_MPI, is_main_process, set_mpi_disabled
from cobaya.tools import warn_deprecation, recursive_update, sort_cosmetic, \
    check_deprecated_modules_path
from cobaya.post import post


def run(info_or_yaml_or_file, packages_path: [str, None] = None,
        output: [str, None] = None, debug: [bool, None] = None,
        stop_at_error: [bool, None] = None, resume=False, force=False,
        no_mpi=False, test=False):
    """
    Run from an input dictionary, file name or yaml string, with optional arguments
    to override settings in the input as needed.

    :param info_or_yaml_or_file: dictionary, yaml file name, or yaml text with options
    :param packages_path: path where external packages were installed
    :param output: path name prefix for output files
    :param debug: verbose debug output
    :param stop_at_error: stop if an error is raised
    :param resume: continue an existing run
    :param force: overwrite existing output if it exists
    :param no_mpi: run without MPI
    :param test: only test initialization rather than actually running
    :return: (updated_info, sampler) tuple of options dictionary and Sampler instance
    """

    # This function reproduces the model-->output-->sampler pipeline one would follow
    # when instantiating by hand, but alters the order to performs checks and dump info
    # as early as possible, e.g. to check if resuming possible or `force` needed.
    if no_mpi or test:
        set_mpi_disabled()
    if isinstance(info_or_yaml_or_file, str):
        if "\n" in info_or_yaml_or_file:
            info = yaml_load(info_or_yaml_or_file)
        else:
            info = load_input_file(info_or_yaml_or_file)
    else:
        assert isinstance(info_or_yaml_or_file, Mapping), (
            "The first argument must be a dictionary, file name or yaml string with the "
            "info needed for the run.")
        info = dict(info_or_yaml_or_file)

    # solve packages installation path cmd > env > input
    if packages_path:
        info[_packages_path] = packages_path
    if debug is not None:
        info[_debug] = bool(debug)
    if test:
        info[_test_run] = True
    if stop_at_error is not None:
        info["stop_at_error"] = bool(stop_at_error)
    # If any of resume|force given as cmd args, ignore those in the input file
    if resume or force:
        if resume and force:
            raise ValueError("'rename' and 'force' are exclusive options")
        info[_resume] = bool(resume)
        info[_force] = bool(force)
    if output:
        info[_output_prefix] = output
    if _post in info:
        post(info)
        return

    logger_setup(info.get(_debug), info.get(_debug_file))
    logger_run = logging.getLogger(__name__.split(".")[-1])
    # MARKED FOR DEPRECATION IN v3.0
    # BEHAVIOUR TO BE REPLACED BY ERROR:
    check_deprecated_modules_path(info)
    # END OF DEPRECATION BLOCK
    # 1. Prepare output driver, if requested by defining an output_prefix
    # GetDist needs to know the original sampler, so don't overwrite if minimizer
    try:
        which_sampler = list(info["sampler"])[0]
    except (KeyError, TypeError):
        raise LoggedError(
            logger_run, "You need to specify a sampler using the 'sampler' key as e.g. "
                        "`sampler: {mcmc: None}.`")
    infix = "minimize" if which_sampler == "minimize" else None
    output = get_output(prefix=info.get(_output_prefix), resume=info.get(_resume),
                        force=info.get(_force), infix=infix)
    # 2. Update the input info with the defaults for each component
    updated_info = update_info(info)
    if logging.root.getEffectiveLevel() <= logging.DEBUG:
        # Dump only if not doing output (otherwise, the user can check the .updated file)
        if not output and is_main_process():
            logger_run.info(
                "Input info updated with defaults (dumped to YAML):\n%s",
                yaml_dump(sort_cosmetic(updated_info)))
    # 3. If output requested, check compatibility if existing one, and dump.
    # 3.1 First: model only
    output.check_and_dump_info(info, updated_info, cache_old=True,
                               ignore_blocks=[kinds.sampler])
    # 3.2 Then sampler -- 1st get the last sampler mentioned in the updated.yaml
    # TODO: ideally, using Minimizer would *append* to the sampler block.
    #       Some code already in place, but not possible at the moment.
    try:
        last_sampler = list(updated_info[kinds.sampler])[-1]
        last_sampler_info = {last_sampler: updated_info[kinds.sampler][last_sampler]}
    except (KeyError, TypeError):
        raise LoggedError(logger_run, "No sampler requested.")
    sampler_name, sampler_class = get_sampler_name_and_class(last_sampler_info)
    check_sampler_info(
        (output.reload_updated_info(use_cache=True) or {}).get(kinds.sampler),
        updated_info[kinds.sampler], is_resuming=output.is_resuming())
    # Dump again, now including sampler info
    output.check_and_dump_info(info, updated_info, check_compatible=False)
    # Check if resumable run
    sampler_class.check_force_resume(
        output, info=updated_info[kinds.sampler][sampler_name])
    # 4. Initialize the posterior and the sampler
    with Model(updated_info[_params], updated_info[kinds.likelihood],
               updated_info.get(_prior), updated_info.get(kinds.theory),
               packages_path=info.get(_packages_path), timing=updated_info.get(_timing),
               allow_renames=False, stop_at_error=info.get("stop_at_error", False)) \
            as model:
        # Re-dump the updated info, now containing parameter routes and version info
        updated_info = recursive_update(updated_info, model.info())
        output.check_and_dump_info(None, updated_info, check_compatible=False)
        sampler = sampler_class(updated_info[kinds.sampler][sampler_name],
                                model, output, packages_path=info.get(_packages_path))
        # Re-dump updated info, now also containing updates from the sampler
        updated_info[kinds.sampler][sampler.get_name()] = \
            recursive_update(
                updated_info[kinds.sampler][sampler.get_name()], sampler.info())
        # TODO -- maybe also re-dump model info, now possibly with measured speeds
        # (waiting until the camb.transfers issue is solved)
        output.check_and_dump_info(None, updated_info, check_compatible=False)
        if info.get(_test_run, False):
            logger_run.info("Test initialization successful! "
                            "You can probably run now without `--%s`.", _test_run)
            return updated_info, sampler
        # Run the sampler
        sampler.run()
    return updated_info, sampler


# Command-line script
def run_script(help_commands=None):
    warn_deprecation()
    import argparse
    parser = argparse.ArgumentParser(
        prog="cobaya run", description="Cobaya's run script.")
    parser.add_argument("input_file", action="store", metavar="input_file.yaml",
                        help="An input file to run.")
    parser.add_argument("-" + _packages_path_arg[0], "--" + _packages_path_arg_posix,
                        action="store", metavar="/packages/path", default=None,
                        help="Path where external packages were installed.")
    # MARKED FOR DEPRECATION IN v3.0
    modules = "modules"
    parser.add_argument("-" + modules[0], "--" + modules,
                        action="store", required=False,
                        metavar="/packages/path", default=None,
                        help="To be deprecated! "
                             "Alias for %s, which should be used instead." %
                             _packages_path_arg_posix)
    # END OF DEPRECATION BLOCK -- CONTINUES BELOW!
    parser.add_argument("-" + _output_prefix[0], "--" + _output_prefix,
                        action="store", metavar="/some/path", default=None,
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
    parser.add_argument("--%s" % _test_run, action="store_true",
                        help="Initialize model and sampler, and exit.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--no-mpi", action='store_true',
                        help="disable MPI when mpi4py installed but MPI does "
                             "not actually work")
    arguments = parser.parse_args()

    # MARKED FOR DEPRECATION IN v3.0
    if arguments.modules is not None:
        logger_setup()
        logger = logging.getLogger(__name__.split(".")[-1])
        logger.warning("*DEPRECATION*: -m/--modules will be deprecated in favor of "
                       "-%s/--%s in the next version. Please, use that one instead.",
                       _packages_path_arg[0], _packages_path_arg_posix)
        # BEHAVIOUR TO BE REPLACED BY ERROR:
        if getattr(arguments, _packages_path_arg) is None:
            setattr(arguments, _packages_path_arg, arguments.modules)
    del arguments.modules
    # END OF DEPRECATION BLOCK
    info = load_input_file(arguments.input_file,
                           no_mpi=arguments.no_mpi or arguments.test,
                           help_commands=help_commands)
    del arguments.input_file
    run(info, **arguments.__dict__)


def load_input_file(input_file, no_mpi=False, help_commands: [str, None] = None):
    if no_mpi:
        set_mpi_disabled()
    if any((os.path.splitext(f)[0] in ("input", "updated"))
           for f in input_file):
        raise ValueError("'input' and 'updated' are reserved file names. "
                         "Please, use a different one.")
    load_input = import_MPI(".input", "load_input")
    if any(input_file.lower().endswith(ext) for ext in _yaml_extensions):
        info = load_input(input_file)
    else:
        # Passed an existing output_prefix? Try to find the corresponding *.updated.yaml
        updated_file = get_info_path(*split_prefix(input_file), kind="updated")
        try:
            info = load_input(updated_file)
        except IOError:
            err_msg = "Not a valid input file, or non-existent run to resume."
            if help_commands:
                err_msg += (" Maybe you mistyped one of the following commands: "
                            + help_commands)
            raise ValueError(err_msg)
        # We need to update the output_prefix to resume the run *where it is*
        info[_output_prefix] = input_file
        # If input given this way, we obviously want to resume!
        info[_resume] = True
    return info
