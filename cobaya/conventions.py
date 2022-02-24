"""
.. module:: conventions

:Synopsis: Some physical constants and naming conventions
           to make the life of the maintainer easier.
:Author: Jesus Torrado

"""

# Package name (for importlib)
# (apparently __package__ is only defined if you import something locally.
cobaya_package = __name__.rpartition('.')[0]


def get_version():
    from cobaya import __version__
    return __version__


debug_default = False
resume_default = False

kinds = ("sampler", "theory", "likelihood")

# Reserved attributes for component classes with defaults.
# These are ignored by HasDefaults.get_class_options()
reserved_attributes = {"input_params", "output_params", "install_options",
                       "bibtex_file", "file_base_name"}

# Conventional order for yaml dumping (purely cosmetic)
dump_sort_cosmetic = ["theory", "likelihood", "prior", "params", "sampler", "post"]

# Separator for
# fields in parameter names and files
# Its manual inclusion in a string anywhere else (e.g. a parameter name) should be avoided
derived_par_name_separator = "__"
separator_files = "."


# Names for the samples' fields internally and in the output
class OutPar:
    weight = "weight"  # sample weight
    # minus log-posterior, or in general the total minus log-probability
    minuslogpost = "minuslogpost"
    minuslogprior = "minuslogprior"  # minus log-prior
    chi2 = "chi2"  # chi^2 = -2 * loglike (not always normalized to be useful)


prior_1d_name = "0"


def get_chi2_name(p):
    return OutPar.chi2 + derived_par_name_separator + str(p)


def undo_chi2_name(p):
    return p[len(OutPar.chi2 + derived_par_name_separator):]


def get_chi2_label(p):
    return r"\chi^2_\mathrm{" + str(p).replace("_", r"\ ") + "}"


def get_minuslogpior_name(piname):
    return OutPar.minuslogprior + derived_par_name_separator + piname


def minuslogprior_names(prior):
    return [get_minuslogpior_name(piname) for piname in prior]


def chi2_names(likes):
    return [get_chi2_name(likname) for likname in likes]


# Output files

class FileSuffix:
    input = "input"
    updated = "updated"


class Extension:
    yaml = ".yaml"
    yamls = ".yaml", ".yml"
    dill = ".dill_pickle"
    checkpoint = ".checkpoint"
    progress = ".progress"
    covmat = ".covmat"
    evidence = ".logZ"


# Installation and container definitions
packages_path_arg = "packages_path"
packages_path_input = packages_path_arg
packages_path_arg_posix = packages_path_arg.replace("_", "-")
packages_path_env = "COBAYA_PACKAGES_PATH"
packages_path_config_file = "config.yaml"
packages_path_containers = "/cobaya_packages"

install_skip_env = "COBAYA_INSTALL_SKIP"
test_skip_env = "COBAYA_TEST_SKIP"

products_path = "/products"

data_path = "data"
code_path = "code"

# Internal package structure
subfolders = {"likelihood": "likelihoods",
              "sampler": "samplers",
              "theory": "theories"}

# Approximate overhead of cobaya per posterior evaluation. Useful for blocking speeds
overhead_time = 0.0003

# Line width for console printing
line_width = 120


# Physical constants
# ------------------
# Light speed
class Const:
    c_km_s = 299792.458  # speed of light
    h_J_s = 6.626070040e-34  # Planck's constant
    kB_J_K = 1.38064852e-23  # Boltzmann constant
