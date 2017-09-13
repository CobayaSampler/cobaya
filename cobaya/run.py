"""
.. module:: run

:Synopsis: Main scripts to run the sampling
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
import six


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
        from cobaya.yaml_custom import yaml_custom_dump
        logging.getLogger(__name__).debug(
            "Input (dumped to YAML):\n%s", yaml_custom_dump(info))
    
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

    # Initialise parametrisation, likelihoods and prior
#    with Parametrisation(info[_params)
    with Likelihood(
            info_likelihood=info[_likelihood], info_params=info[_params],
            info_prior=info.get(_prior), info_theory=info.get(_theory),
            path_to_installation=info.get(_path_install)) as likelihood:
        with Prior(likelihood.sampled_params(), info_prior=likelihood.updated_info_prior()) as prior:
            with Sampler(info[_sampler], prior, likelihood, output) as sampler:
                # Save the model info (updated by the likelihood and prior)
                output.update_info(
                    likelihood=likelihood.updated_info(),
                    theory=likelihood.updated_info_theory(),
                    params=likelihood.updated_info_params(),
                    prior=likelihood.updated_info_prior(),
                    sampler=sampler.updated_info())
                output.dump_info()
                # Sample!
                sampler.run()
    # For scripted calls
    return output.updated_info(), sampler.products()

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
        info[input_path_auto] = path_cmd
    run(info)
