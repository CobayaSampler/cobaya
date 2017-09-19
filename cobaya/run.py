"""
.. module:: run

:Synopsis: Main scripts to run the sampling
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
import six

# Global
from copy import deepcopy


def run(info):

    assert hasattr(info, "items"), (
        "The agument of `run` must be a dictionary with the info needed for the run. "
        "If you were trying to pass an input file instead, load it first with "
        "`cobaya.input.load_input`.")

    # Import names
    from cobaya.conventions import _likelihood, _prior, _params
    from cobaya.conventions import _theory, _sampler, _path_install
    from cobaya.conventions import _debug, _debug_file, _output_prefix

    # Configure the logger ASAP
    from cobaya.log import logger_setup
    logger_setup(info.get(_debug), info.get(_debug_file))
    
    # Debug (lazy call)
    import logging
    if logging.root.getEffectiveLevel() <= logging.DEBUG:
        from cobaya.yaml_custom import yaml_dump
        logging.getLogger(__name__).debug(
            "Input info (dumped to YAML):\n%s", yaml_dump(info))
    
    # Import general classes
    from cobaya.prior import Prior
    from cobaya.sampler import get_Sampler as Sampler

    # Import the functions and classes that need MPI wrapping
    from cobaya.mpi import import_MPI
#    Likelihood = import_MPI(".likelihood", "LikelihoodCollection")
    from cobaya.likelihood import LikelihoodCollection as Likelihood

    # Initialise output, if requiered
    do_output = info.get(_output_prefix)
    if do_output:
        Output = import_MPI(".output", "Output")
        output = Output(info)
    else:
        from cobaya.output import Output_dummy
        output = Output_dummy(info)

    # Create the full input information, including defaults for each module
    from cobaya.input import get_full_info
    full_info = get_full_info(info)
    output.set_full_info(full_info)
    if logging.root.getEffectiveLevel() <= logging.DEBUG:
        logging.getLogger(__name__).debug(
            "Updated info (dumped to YAML):\n%s", yaml_dump(full_info))

    # Set the path of the installed modules, if given
    from cobaya.tools import set_path_to_installation
    set_path_to_installation(info.get(_path_install))

    # Initialise parametrisation, likelihoods and prior
    from cobaya.parametrisation import Parametrisation
    with Parametrisation(full_info[_params]) as par:
        with Prior(par, full_info.get(_prior)) as prior:
            with Likelihood(full_info[_likelihood], par, full_info.get(_theory)) as lik:
                with Sampler(full_info[_sampler], par, prior, lik, output) as sampler:
                    # Save the full, defaults-populated info
                    output.dump_info()
                    # Sample!
                    sampler.run()
    # For scripted calls
    return deepcopy(full_info), sampler.products()


# Command-line script
def run_script():
    from cobaya.mpi import import_MPI
    load_input = import_MPI(".input", "load_input")
    import argparse
    parser = argparse.ArgumentParser(description="Cobaya's run script.")
    parser.add_argument("input_file", nargs=1, action="store", metavar="input_file.yaml",
                        help="An input file to run.")
    parser.add_argument("-p", "--path",
                        action="store", nargs="+", metavar=("/some/path"),
                        help="Path where modules were automatically installed.")
    args = parser.parse_args()
    info = load_input(args.input_file[0])
    # solve path
    from cobaya.conventions import _path_install
    path_cmd = (lambda x: x[0] if x else None)(getattr(args, "path"))
    path_input = info.get(_path_install)
    if path_cmd:
        if path_input:
            raise ValueError("CONFLICT: "
                      "You have specified a modules folder both in the command line "
                      "('%s') and the input file ('%s'). There should only be one."%(
                          path_cmd, path_input))
        info[_path_install] = path_cmd
    run(info)
