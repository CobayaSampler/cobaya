"""
.. module:: run

:Synopsis: Main scripts to run the sampling
:Author: Jesus Torrado

"""

import os

from cobaya import mpi
from cobaya.conventions import (
    get_version,
    packages_path_arg,
    packages_path_arg_posix,
    packages_path_input,
)
from cobaya.input import load_info_overrides, update_info
from cobaya.log import LoggedError, get_logger, is_debug, logger_setup
from cobaya.model import Model
from cobaya.output import get_output
from cobaya.post import PostResult, post
from cobaya.sampler import Sampler, check_sampler_info, get_sampler_name_and_class
from cobaya.tools import recursive_update, sort_cosmetic, warn_deprecation
from cobaya.typing import InputDict, LiteralFalse
from cobaya.yaml import yaml_dump


def run(
    info_or_yaml_or_file: InputDict | str | os.PathLike,
    packages_path: str | None = None,
    output: str | LiteralFalse | None = None,
    debug: bool | int | None = None,
    stop_at_error: bool | None = None,
    resume: bool | None = None,
    force: bool | None = None,
    minimize: bool | None = None,
    no_mpi: bool = False,
    test: bool | None = None,
    override: InputDict | None = None,
    allow_changes: bool = False,
) -> tuple[InputDict, Sampler | PostResult]:
    """
    Run from an input dictionary, file name or yaml string, with optional arguments
    to override settings in the input as needed.

    :param info_or_yaml_or_file: input options dictionary, yaml file, or yaml text
    :param packages_path: path where external packages were installed
    :param output: path name prefix for output files, or False for no file output
    :param debug: true for verbose debug output, or a specific logging level
    :param stop_at_error: stop if an error is raised
    :param resume: continue an existing run
    :param force: overwrite existing output if it exists
    :param minimize: if true, ignores the sampler and runs default minimizer
    :param no_mpi: run without MPI
    :param test: only test initialization rather than actually running
    :param override: option dictionary to merge into the input one, overriding settings
       (but with lower precedence than the explicit keyword arguments)
    :param allow_changes: if true, allow input option changes when resuming or minimizing
    :return: (updated_info, sampler) tuple of options dictionary and Sampler instance,
              or (updated_info, post_results) if using "post" post-processing
    """
    # This function reproduces the model-->output-->sampler pipeline one would follow
    # when instantiating by hand, but alters the order to perform checks and dump info
    # as early as possible, e.g. to check if resuming possible or `force` needed.
    if no_mpi or test:
        mpi.set_mpi_disabled()
    with mpi.ProcessState("run"):
        flags = {
            packages_path_input: packages_path,
            "debug": debug,
            "stop_at_error": stop_at_error,
            "resume": resume,
            "force": force,
            "minimize": minimize,
            "test": test,
        }
        info: InputDict = load_info_overrides(
            info_or_yaml_or_file, override or {}, **flags
        )
        logger_setup(info.get("debug"))
        logger_run = get_logger(run.__name__)
        try:
            which_sampler = list(info["sampler"])[0]
            if info.get("minimize"):  # type: ignore
                # Preserve options if "minimize" was already the sampler
                if which_sampler.lower() != "minimize":
                    info["sampler"] = {"minimize": None}
                    which_sampler = "minimize"
        except (KeyError, TypeError) as excpt:
            if not info.get("post"):
                raise LoggedError(
                    logger_run,
                    "You need to specify a sampler using the 'sampler' key "
                    "as e.g. `sampler: {mcmc: None}.`",
                ) from excpt
        if info.get("post"):
            if isinstance(output, str) or output is False:
                info["post"]["output"] = output or None
            return post(info)
        # Set up output and logging
        if isinstance(output, str) or output is False:
            info["output"] = output or None
        # 1. Prepare output driver, if requested by defining an output prefix
        infix = "minimize" if which_sampler == "minimize" else None
        with get_output(
            prefix=info.get("output"),
            resume=info.get("resume"),
            force=info.get("force"),
            infix=infix,
        ) as out:
            # 2. Update the input info with the defaults for each component
            updated_info = update_info(info)
            if is_debug(logger_run):
                # Dump only if not doing output
                # (otherwise, the user can check the .updated file)
                if not out and mpi.is_main_process():
                    logger_run.info(
                        "Input info updated with defaults (dumped to YAML):\n%s",
                        yaml_dump(sort_cosmetic(updated_info)),
                    )
            # 3. If output requested, check compatibility if existing one, and dump.
            # 3.1 First: model only
            out.check_and_dump_info(
                info,
                updated_info,
                cache_old=True,
                check_compatible=not allow_changes,
                ignore_blocks=["sampler"],
            )
            # 3.2 Then sampler -- 1st get the first sampler mentioned in the updated.yaml
            if not (info_sampler := updated_info.get("sampler")):
                raise LoggedError(logger_run, "No sampler requested.")
            sampler_name, sampler_class = get_sampler_name_and_class(info_sampler)
            updated_info["sampler"] = check_sampler_info(
                (out.get_updated_info(use_cache=True) or {}).get("sampler"),
                updated_info["sampler"],
                is_resuming=out.is_resuming(),
            )
            # Dump again, now including sampler info
            out.check_and_dump_info(info, updated_info, check_compatible=False)
            # Check if resumable run
            sampler_class.check_force_resume(
                out, info=updated_info["sampler"][sampler_name]
            )
            # 4. Initialize the posterior and the sampler
            with Model(
                updated_info["params"],
                updated_info["likelihood"],
                updated_info.get("prior"),
                updated_info.get("theory"),
                packages_path=info.get("packages_path"),
                timing=updated_info.get("timing"),
                allow_renames=False,
                stop_at_error=info.get("stop_at_error", False),
            ) as model:
                # Re-dump the updated info, now containing parameter routes and version
                updated_info = recursive_update(updated_info, model.info())
                out.check_and_dump_info(None, updated_info, check_compatible=False)
                sampler = sampler_class(
                    updated_info["sampler"][sampler_name],
                    model,
                    out,
                    name=sampler_name,
                    packages_path=info.get("packages_path"),
                )
                # Re-dump updated info, now also containing updates from the sampler
                updated_info["sampler"][sampler_name] = recursive_update(
                    updated_info["sampler"][sampler_name], sampler.info()
                )
                out.check_and_dump_info(None, updated_info, check_compatible=False)
                mpi.sync_processes()
                if info.get("test"):
                    logger_run.info(
                        "Test initialization successful! "
                        "You can probably run now without `--%s`.",
                        "test",
                    )
                    return updated_info, sampler
                # Run the sampler
                sampler.run()
    return updated_info, sampler


# Command-line script
def run_script(args=None):
    """Shell script wrapper for :func:`run.run` (including :func:`post.post`)"""
    warn_deprecation()
    import argparse

    # kwargs for flags that should be True|None, instead of True|False
    # (needed in order not to mistakenly override input file inside the run() function
    trueNone_kwargs = {"action": "store_true", "default": None}
    parser = argparse.ArgumentParser(
        prog="cobaya run", description="Cobaya's run script."
    )
    parser.add_argument(
        "input_file",
        action="store",
        metavar="input_file.yaml",
        help="An input file to run.",
    )
    parser.add_argument(
        "-" + packages_path_arg[0],
        "--" + packages_path_arg_posix,
        action="store",
        metavar="/packages/path",
        default=None,
        help="Path where external packages were installed.",
    )
    parser.add_argument(
        "-" + "o",
        "--" + "output",
        action="store",
        metavar="/some/path",
        default=None,
        help="Path and prefix for the text output.",
    )
    parser.add_argument(
        "-" + "d", "--" + "debug", help="Produce verbose debug output.", **trueNone_kwargs
    )
    continuation = parser.add_mutually_exclusive_group(required=False)
    continuation.add_argument(
        "-" + "r",
        "--" + "resume",
        help=("Resume an existing chain if it has similar info (fails otherwise)."),
        **trueNone_kwargs,
    )
    continuation.add_argument(
        "-" + "f",
        "--" + "force",
        help=("Overwrites previous output, if it exists (use with care!)"),
        **trueNone_kwargs,
    )
    parser.add_argument(
        "--%s" % "test", help="Initialize model and sampler, and exit.", **trueNone_kwargs
    )
    parser.add_argument(
        "-M",
        "--minimize",
        help=(
            "Replaces the sampler in the input and runs a minimization "
            "process (incompatible with post-processing)."
        ),
        **trueNone_kwargs,
    )
    parser.add_argument(
        "--allow-changes",
        help="Allow changing input parameters when resuming "
        "or minimizing, skipping consistency checks",
        **trueNone_kwargs,
    )
    parser.add_argument("--version", action="version", version=get_version())
    parser.add_argument(
        "--no-mpi",
        help=("disable MPI when mpi4py installed but MPI does not actually work"),
        **trueNone_kwargs,
    )
    arguments = parser.parse_args(args)
    info = arguments.input_file
    del arguments.input_file
    if not info.endswith(".yaml") and not os.path.exists(info):
        if os.path.exists(info + ".yaml"):
            info = info + ".yaml"
    run(info, **arguments.__dict__)


if __name__ == "__main__":
    run_script()
