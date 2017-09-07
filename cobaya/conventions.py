"""
.. module:: conventions

:Synopsis: Some naming conventions to make the life of the maintainer easier.
:Author: Jesus Torrado

"""
# Package name (for importlib)
# (apparently __package__ is only defined if you import something locally.
package = __name__.rpartition('.')[0]

# Names for block and fields in the input
input_likelihood = "likelihood"
input_prior      = "prior"
input_theory     = "theory"
input_sampler    = "sampler"
input_params     = "params"
input_p_dist     = "dist"
input_p_ref      = "ref"
input_p_label    = "latex"
input_debug      = "debug"
input_debug_file = "debug_file"
input_output_prefix = "output_prefix"
input_path_install = "path_to_modules"
input_likelihood_external = "external"

# Separator for different names.
# Its manual inclusion in a string anywhere else (e.g. a parameter name) is forbidden.
separator = "__"

# Names for the samples' fields internally and in the output
_weight        = "weight"        # sample weight
_minuslogpost  = "minuslogpost"  # log-posterior, or in general the total log-probability
_minuslogprior = "minuslogprior" # log-prior
_chi2          = "chi2"          # chi^2 = -2 * loglik
_derived_pre   = "derived"+separator  # prefix for derived parameters

# Output files
output_input_suffix = "input"
output_full_suffix  = "full"

# Internal package structure
defaults_file = "defaults.yaml"
subfolders = {input_likelihood: "likelihoods",
              input_sampler: "samplers",
              input_theory: "theories"}
