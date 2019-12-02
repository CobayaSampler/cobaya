"""
.. module:: conventions

:Synopsis: Some physical constants and naming conventions
           to make the life of the maintainer easier.
:Author: Jesus Torrado

"""
from collections import namedtuple

# Package name (for importlib)
# (apparently __package__ is only defined if you import something locally.
_package = __name__.rpartition('.')[0]

# Names for block and fields in the input
_prior = "prior"
_post = "post"
_post_add = "add"
_post_remove = "remove"
_post_suffix = "suffix"
_params = "params"
_input_params = "input_params"
_output_params = "output_params"
_input_params_prefix = "input_params_prefix"
_output_params_prefix = "output_params_prefix"
_debug = "debug"
_debug_default = False
_debug_file = "debug_file"
_output_prefix = "output"
_path_install = "modules"
_external = "external"
_provides = "provides"
_requires = "requires"
_resume = "resume"
_resume_default = False
_timing = "timing"
_force = "force"
_module_path = "python_path"
_module_class_name = "class_name"
_aliases = "aliases"
_version = "version"

ParTags = namedtuple('ParTags', ("prior", "ref", "proposal", "value", "dist", "drop",
                                 "derived", "latex", "renames"))
partag = ParTags(*ParTags._fields)

ComponentKinds = namedtuple('ComponentKinds', ("sampler", "theory", "likelihood"))
kinds = ComponentKinds(*ComponentKinds._fields)

# Separator for
# fields in parameter names and files
# Its manual inclusion in a string anywhere else (e.g. a parameter name) should be avoided
_separator = "__"
_separator_files = "."

# Names for the samples' fields internally and in the output
_weight = "weight"  # sample weight
_minuslogpost = "minuslogpost"  # log-posterior, or in general the total log-probability
_minuslogprior = "minuslogprior"  # log-prior
_chi2 = "chi2"  # chi^2 = -2 * loglik
_prior_1d_name = "0"

# Output files
_input_suffix = "input"
_updated_suffix = "updated"
_yaml_extensions = ".yaml", ".yml"
_checkpoint_extension = ".checkpoint"
_progress_extension = ".progress"
_covmat_extension = ".covmat"

# Installation and container definitions
_modules_path_arg = _path_install
_modules_path_env = "COBAYA_MODULES"
_modules_path = "/modules"
_code = "code"
_data = "data"
_covmats_file = "covmats_database.yaml"
_products_path = "/products"
_requirements_file = "requirements.yaml"
_help_file = "readme.md"

# Internal package structure
subfolders = {kinds.likelihood: "likelihoods",
              kinds.sampler: "samplers",
              kinds.theory: "theories"}

# Approximate overhead of cobaya per posterior evaluation. Useful for blocking speeds
# After testing, it's mostly due to evaluating logpdf of scipy.stats 1d pdf's,
# in particular ~0.1ms per param
# (faster for most common cases, after manual override of logpdf)
_overhead_per_param = 5e-6

# Line width for console printing
_line_width = 120

# Physical constants
# ------------------
# Light speed
_c_km_s = 299792.458  # speed of light
_h_J_s = 6.626070040e-34  # Planck's constant
_kB_J_K = 1.38064852e-23  # Boltzmann constant
