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
    from cobaya.conventions import input_likelihood, input_prior, input_params
    from cobaya.conventions import input_theory, input_sampler
    from cobaya.conventions import input_debug, input_debug_file, input_output_prefix

    # Configure the logger ASAP
    from cobaya.log import logger_setup
    logger_setup(info.get(input_debug), info.get(input_debug_file))
    
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
    do_output = info.get(input_output_prefix)
    if do_output:
        Output = import_MPI(".output", "Output")
        output = Output(info)
    else:
        from cobaya.output import Output_dummy
        output = Output_dummy(info)

    # Initialise likelihoods and prior
    with Likelihood(
            info[input_likelihood], info[input_params], info_prior=info.get(input_prior),
            info_theory=info.get(input_theory)) as likelihood:
        with Prior(likelihood.sampled_params(), info_prior=likelihood.updated_info_prior()) as prior:
            with Sampler(info[input_sampler], prior, likelihood, output) as sampler:
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
    import sys
    try:
        input_file = sys.argv[1]
    except IndexError:
        raise IndexError("You must provide an input yaml file as an argument.")
    info = load_input(input_file)
    run(info)

