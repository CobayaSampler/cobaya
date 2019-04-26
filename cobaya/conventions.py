"""
.. module:: conventions

:Synopsis: Some physical constants and naming conventions
           to make the life of the maintainer easier.
:Author: Jesus Torrado

"""
# Package name (for importlib)
# (apparently __package__ is only defined if you import something locally.
_package = __name__.rpartition('.')[0]

# Names for block and fields in the input
_likelihood = "likelihood"
_prior = "prior"
_theory = "theory"
_sampler = "sampler"
_params = "params"
_p_value = "value"
_p_dist = "dist"
_p_ref = "ref"
_p_label = "latex"
_p_renames = "renames"
_p_drop = "drop"
_p_derived = "derived"
_p_proposal = "proposal"
_debug = "debug"
_debug_default = False
_debug_file = "debug_file"
_output_prefix = "output"
_path_install = "modules"
_external = "external"
_resume = "resume"
_resume_default = False
_timing = "timing"
_force = "force"

# Separator for different names.
# Its manual inclusion in a string anywhere else (e.g. a parameter name) is forbidden.
_separator = "__"

# Names for the samples' fields internally and in the output
_weight = "weight"  # sample weight
_minuslogpost = "minuslogpost"  # log-posterior, or in general the total log-probability
_minuslogprior = "minuslogprior"  # log-prior
_chi2 = "chi2"  # chi^2 = -2 * loglik
_prior_1d_name = "0"

# Output files
_input_suffix = "input"
_full_suffix = "full"
_yaml_extensions = ".yaml", ".yml"
_checkpoint_extension = ".checkpoint"
_covmat_extension = ".covmat"

# Installation and container definitions
_modules_path_arg = "modules"
_modules_path = "/modules"
_code = "code"
_data = "data"
_covmats_file = "covmats_database.yaml"
_products_path = "/products"
_requirements_file = "requirements.yaml"
_help_file = "readme.md"

# Internal package structure
subfolders = {_likelihood: "likelihoods",
              _sampler: "samplers",
              _theory: "theories"}

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
_T_CMB_K = 2.72548  # CMB temperature
_h_J_s = 6.626070040e-34  # Planck's constant
_kB_J_K = 1.38064852e-23  # Boltzmann constant
