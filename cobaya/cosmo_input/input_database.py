# Notes about writing more presets:
# ---------------------------------
# - all parameter names below are PLANCK parameter names. They are substituted by the
#   theory-code-specific ones in `create_input`
# - don't use extra_args for precision parameters! because if the same precision param
#   is mentioned twice at the same time in different fields with different values, there
#   is no facility to take the max (or min). Instead, codify precision needs in terms of
#   requirements in the .must_provide method of the cosmo code.

from copy import deepcopy

from cobaya.typing import InfoDict

none = "(None)"
error_msg = "error_msg"

# Theory codes
theory: InfoDict = {"camb": None, "classy": None}

# Primordial perturbations
primordial = {
    "SFSR": {
        "desc": "Adiabatic scalar perturbations, power law spectrum",
        "theory": theory,
        "params": {
            "logA": {
                "prior": {"min": 1.61, "max": 3.91},
                "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
                "proposal": 0.001,
                "latex": "\\log(10^{10} A_\\mathrm{s})",
                "drop": True,
            },
            "As": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": "A_\\mathrm{s}"},
            "ns": {
                "prior": {"min": 0.8, "max": 1.2},
                "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
                "proposal": 0.002,
                "latex": "n_\\mathrm{s}",
            },
        },
    },
    "SFSR_DESpriors": {
        "desc": "Adiabatic scalar perturbations, power law - DESpriors",
        "theory": theory,
        "params": {
            "As_1e9": {
                "prior": {"min": 0.5, "max": 5},
                "ref": {"dist": "norm", "loc": 2.1, "scale": 0.5},
                "proposal": 0.25,
                "latex": "10^9 A_\\mathrm{s})",
                "drop": True,
                "renames": "A",
            },
            "As": {"value": "lambda As_1e9: 1e-9 * As_1e9", "latex": "A_\\mathrm{s}"},
            "ns": {
                "prior": {"min": 0.87, "max": 1.07},
                "ref": {"dist": "norm", "loc": 0.965, "scale": 0.05},
                "proposal": 0.002,
                "latex": "n_\\mathrm{s}",
            },
        },
    },
    "SFSR_lenspriors": {
        "desc": "Adiabatic scalar perturbations, power law -- Planck lensing priors",
        "theory": theory,
        "params": {
            "logA": {
                "prior": {"min": 1.61, "max": 3.91},
                "ref": {"dist": "norm", "loc": 3.05, "scale": 0.01},
                "proposal": 0.005,
                "latex": "\\log(10^{10} A_\\mathrm{s})",
                "drop": True,
            },
            "As": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": "A_\\mathrm{s}"},
            "ns": {
                "prior": {"dist": "norm", "loc": 0.96, "scale": 0.02},
                "ref": {"dist": "norm", "loc": 0.965, "scale": 0.02},
                "proposal": 0.002,
                "latex": "n_\\mathrm{s}",
            },
        },
    },
    "SFSR_run": {
        "desc": "Adiabatic scalar perturbations, power law + running spectrum",
        "theory": theory,
        "params": {
            "logA": {
                "prior": {"min": 1.61, "max": 3.91},
                "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
                "proposal": 0.001,
                "latex": "\\log(10^{10} A_\\mathrm{s})",
                "drop": True,
            },
            "As": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": "A_\\mathrm{s}"},
            "ns": {
                "prior": {"min": 0.8, "max": 1.2},
                "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
                "proposal": 0.002,
                "latex": "n_\\mathrm{s}",
            },
            "nrun": {
                "prior": {"min": -1, "max": 1},
                "ref": {"dist": "norm", "loc": 0, "scale": 0.005},
                "proposal": 0.001,
                "latex": "n_\\mathrm{run}",
            },
        },
    },
    "SFSR_runrun": {
        "desc": "Adiabatic scalar perturbations, power law + 2nd-order running spectrum",
        "theory": theory,
        "params": {
            "logA": {
                "prior": {"min": 1.61, "max": 3.91},
                "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
                "proposal": 0.001,
                "latex": "\\log(10^{10} A_\\mathrm{s})",
                "drop": True,
            },
            "As": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": "A_\\mathrm{s}"},
            "ns": {
                "prior": {"min": 0.8, "max": 1.2},
                "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
                "proposal": 0.002,
                "latex": "n_\\mathrm{s}",
            },
            "nrun": {
                "prior": {"min": -1, "max": 1},
                "ref": {"dist": "norm", "loc": 0, "scale": 0.005},
                "proposal": 0.001,
                "latex": "n_\\mathrm{run}",
            },
            "nrunrun": {
                "prior": {"min": -1, "max": 1},
                "ref": {"dist": "norm", "loc": 0, "scale": 0.002},
                "proposal": 0.001,
                "latex": "n_\\mathrm{run,run}",
            },
        },
    },
    "SFSR_t": {
        "desc": "Adiabatic scalar+tensor perturbations, power law spectrum "
        "(inflation consistency)",
        "theory": {
            "camb": {"extra_args": {"nt": None}},
            "classy": {"extra_args": {"n_t": "scc", "alpha_t": "scc"}},
        },
        "params": {
            "logA": {
                "prior": {"min": 1.61, "max": 3.91},
                "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
                "proposal": 0.001,
                "latex": "\\log(10^{10} A_\\mathrm{s})",
                "drop": True,
            },
            "As": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": "A_\\mathrm{s}"},
            "ns": {
                "prior": {"min": 0.8, "max": 1.2},
                "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
                "proposal": 0.002,
                "latex": "n_\\mathrm{s}",
            },
            "r": {
                "prior": {"min": 0, "max": 3},
                "ref": {"dist": "norm", "loc": 0, "scale": 0.03},
                "proposal": 0.03,
                "latex": "r_{0.05}",
            },
        },
    },
    "SFSR_t_nrun": {
        "desc": "Adiabatic scalar+tensor perturbations, power law + running "
        "spectrum (inflation consistency)",
        "theory": {
            "camb": {"extra_args": {"nt": None}},
            "classy": {"extra_args": {"n_t": "scc", "alpha_t": "scc"}},
        },
        "params": {
            "logA": {
                "prior": {"min": 1.61, "max": 3.91},
                "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
                "proposal": 0.001,
                "latex": "\\log(10^{10} A_\\mathrm{s})",
                "drop": True,
            },
            "As": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": "A_\\mathrm{s}"},
            "ns": {
                "prior": {"min": 0.8, "max": 1.2},
                "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
                "proposal": 0.002,
                "latex": "n_\\mathrm{s}",
            },
            "nrun": {
                "prior": {"min": -1, "max": 1},
                "ref": {"dist": "norm", "loc": 0, "scale": 0.005},
                "proposal": 0.001,
                "latex": "n_\\mathrm{run}",
            },
            "r": {
                "prior": {"min": 0, "max": 3},
                "ref": {"dist": "norm", "loc": 0, "scale": 0.03},
                "proposal": 0.03,
                "latex": "r_{0.05}",
            },
        },
    },
}

# Geometry
geometry = {
    "flat": {"desc": "Flat FLRW universe", "theory": theory},
    "omegak": {
        "desc": "FLRW model with varying curvature (prior on Omega_k)",
        "theory": theory,
        "params": {
            "omegak": {
                "prior": {"min": -0.3, "max": 0.3},
                "ref": {"dist": "norm", "loc": -0.009, "scale": 0.001},
                "proposal": 0.001,
                "latex": "\\Omega_k",
            }
        },
    },
}

# Hubble parameter constraints
H0_min, H0_max = 20, 100

hubble = {
    "H": {
        "desc": "Hubble parameter",
        "theory": theory,
        "params": {
            "H0": {
                "prior": {"min": H0_min, "max": H0_max},
                "ref": {"dist": "norm", "loc": 67, "scale": 2},
                "proposal": 2,
                "latex": "H_0",
            }
        },
    },
    "H_DESpriors": {
        "desc": "Hubble parameter (reduced range for DES and lensing-only constraints)",
        "theory": theory,
        "params": {
            "H0": {
                "prior": {"min": 55, "max": 91},
                "ref": {"dist": "norm", "loc": 67, "scale": 2},
                "proposal": 2,
                "latex": "H_0",
            }
        },
    },
    "sound_horizon_last_scattering": {
        "desc": "Angular size of the sound horizon at last scattering "
        "(approximate, if using CAMB)",
        "theory": {
            "camb": {
                "params": {
                    "theta_MC_100": {
                        "prior": {"min": 0.5, "max": 10},
                        "ref": {"dist": "norm", "loc": 1.04109, "scale": 0.0004},
                        "proposal": 0.0002,
                        "latex": "100\\theta_\\mathrm{MC}",
                        "drop": True,
                        "renames": "theta",
                    },
                    "cosmomc_theta": {
                        "value": "lambda theta_MC_100: 1.e-2*theta_MC_100",
                        "derived": False,
                    },
                    "H0": {"latex": "H_0", "min": H0_min, "max": H0_max},
                },
                "extra_args": {"theta_H0_range": [H0_min, H0_max]},
            },
            "classy": {
                "params": {
                    "theta_s_100": {
                        "prior": {"min": 0.5, "max": 10},
                        "ref": {"dist": "norm", "loc": 1.0416, "scale": 0.0004},
                        "proposal": 0.0002,
                        "latex": "100\\theta_\\mathrm{s}",
                    },
                    "H0": {"latex": "H_0"},
                }
            },
        },
    },
    "sound_horizon_lensonly": {
        "desc": "Angular size of the sound horizon (h>0.4; approximate, if using CAMB)",
        "theory": {
            "camb": {
                "params": {
                    "theta_MC_100": {
                        "prior": {"min": 0.5, "max": 10},
                        "ref": {"dist": "norm", "loc": 1.04109, "scale": 0.002},
                        "proposal": 0.001,
                        "latex": "100\\theta_\\mathrm{MC}",
                        "drop": True,
                        "renames": "theta",
                    },
                    "cosmomc_theta": {
                        "value": "lambda theta_MC_100: 1.e-2*theta_MC_100",
                        "derived": False,
                    },
                    "H0": {"latex": "H_0", "min": 40, "max": H0_max},
                },
                "extra_args": {"theta_H0_range": [40, H0_max]},
            },
        },
    },
}

# Matter sector (minus light species)
N_eff_std = 3.044
nu_mass_fac = 94.0708
matter: InfoDict = {
    "omegab_h2, omegac_h2": {
        "desc": "Flat prior on Omega*h^2 for baryons and cold dark matter",
        "theory": theory,
        "params": {
            "omegabh2": {
                "prior": {"min": 0.005, "max": 0.1},
                "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0001},
                "proposal": 0.0001,
                "latex": "\\Omega_\\mathrm{b} h^2",
            },
            "omegach2": {
                "prior": {"min": 0.001, "max": 0.99},
                "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001},
                "proposal": 0.0005,
                "latex": "\\Omega_\\mathrm{c} h^2",
            },
            "omegam": {"latex": "\\Omega_\\mathrm{m}"},
        },
    },
    "Omegab, Omegam": {
        "desc": "Flat prior on Omega for baryons and total matter",
        "theory": theory,
        "params": {
            "omegab": {
                "prior": {"min": 0.03, "max": 0.07},
                "ref": {"dist": "norm", "loc": 0.0495, "scale": 0.004},
                "proposal": 0.004,
                "latex": "\\Omega_\\mathrm{b}",
                "drop": True,
            },
            "omegam": {
                "prior": {"min": 0.1, "max": 0.9},
                "ref": {"dist": "norm", "loc": 0.316, "scale": 0.02},
                "proposal": 0.02,
                "latex": "\\Omega_\\mathrm{m}",
                "drop": True,
            },
            "omegabh2": {
                "value": "lambda omegab, H0: omegab*(H0/100)**2",
                "latex": "\\Omega_\\mathrm{b} h^2",
            },
            "omegach2": {
                "value": "lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2"
                "-(mnu*(%g/3)**0.75)/%g" % (N_eff_std, nu_mass_fac),
                "latex": "\\Omega_\\mathrm{c} h^2",
            },
        },
    },
    "omegab_h2_lenspriors": {
        "desc": "BBN-like prior on Omega*h^2 for baryons, with cold dark matter",
        "theory": theory,
        "params": {
            "omegabh2": {
                "prior": {"dist": "norm", "loc": 0.0222, "scale": 0.0005},
                "ref": {"dist": "norm", "loc": 0.0222, "scale": 0.0005},
                "proposal": 0.0004,
                "latex": "\\Omega_\\mathrm{b} h^2",
            },
            "omegach2": {
                "prior": {"min": 0.001, "max": 0.99},
                "ref": {"dist": "norm", "loc": 0.12, "scale": 0.003},
                "proposal": 0.002,
                "latex": "\\Omega_\\mathrm{c} h^2",
            },
            "omegam": {"latex": "\\Omega_\\mathrm{m}"},
        },
    },
}

for m in matter.values():
    m["params"]["omegamh2"] = {
        "derived": "lambda omegam, H0: omegam*(H0/100)**2",
        "latex": r"\Omega_\mathrm{m} h^2",
    }

# Neutrinos and other extra matter
neutrinos: InfoDict = {
    "one_heavy_planck": {
        "desc": "Two massless nu and one with m=0.06. Neff=3.044",
        "theory": {
            "camb": {
                "extra_args": {"num_massive_neutrinos": 1, "nnu": 3.044},
                "params": {"mnu": 0.06},
            },
            "classy": {
                "extra_args": {"N_ncdm": 1, "N_ur": 2.0328},
                "params": {"m_ncdm": {"value": 0.06, "renames": "mnu"}},
            },
        },
    },
    "varying_mnu": {
        "desc": "Varying total mass of 3 degenerate nu's, with N_eff=3.044",
        "theory": {
            "camb": {
                "extra_args": {"num_massive_neutrinos": 3, "nnu": 3.044},
                "params": {
                    "mnu": {
                        "prior": {"min": 0, "max": 5},
                        "ref": {"dist": "norm", "loc": 0.02, "scale": 0.1},
                        "proposal": 0.03,
                        "latex": "\\sum m_\\nu",
                    }
                },
            },
            "classy": {
                "extra_args": {"N_ncdm": 1, "deg_ncdm": 3, "N_ur": 0.00641},
                "params": {
                    "m_ncdm": {
                        "prior": {"min": 0, "max": 1.667},
                        "ref": {"dist": "norm", "loc": 0.0067, "scale": 0.033},
                        "proposal": 0.01,
                        "latex": "m_\\nu",
                    },
                    "mnu": {
                        "derived": "lambda m_ncdm: 3 * m_ncdm",
                        "latex": "\\sum m_\\nu",
                    },
                },
            },
        },
    },
    "varying_Neff": {
        "desc": "Varying Neff with two massless nu and one with m=0.06",
        "theory": {
            "camb": {
                "extra_args": {"num_massive_neutrinos": 1},
                "params": {
                    "mnu": 0.06,
                    "nnu": {
                        "prior": {"min": 0.05, "max": 10},
                        "ref": {"dist": "norm", "loc": 3.044, "scale": 0.05},
                        "proposal": 0.05,
                        "latex": "N_\\mathrm{eff}",
                    },
                },
            },
            "classy": {
                "extra_args": {"N_ncdm": 1},
                "params": {
                    "m_ncdm": {"value": 0.06, "renames": "mnu"},
                    "N_ur": {
                        "prior": {"min": 0.0001, "max": 9},
                        "ref": {"dist": "norm", "loc": 2.0328, "scale": 0.05},
                        "proposal": 0.05,
                        "latex": "N_\\mathrm{ur}",
                    },
                    "nnu": {"derived": "lambda Neff: Neff", "latex": "N_\\mathrm{eff}"},
                },
            },
        },
    },
    "varying_mnu_Neff": {
        "desc": "Varying Neff and total mass of 3 degenerate nu's",
        "theory": {
            "camb": {
                "extra_args": {"num_massive_neutrinos": 3},
                "params": {
                    "mnu": {
                        "prior": {"min": 0, "max": 5},
                        "ref": {"dist": "norm", "loc": 0.02, "scale": 0.1},
                        "proposal": 0.03,
                        "latex": "\\sum m_\\nu",
                    },
                    "nnu": {
                        "prior": {"min": 0.05, "max": 10},
                        "ref": {"dist": "norm", "loc": 3.044, "scale": 0.05},
                        "proposal": 0.05,
                        "latex": "N_\\mathrm{eff}",
                    },
                },
            }
        },
    },
}
# Dark Energy
dark_energy: InfoDict = {
    "lambda": {
        "desc": "Cosmological constant (w=-1)",
        "theory": theory,
        "params": {"omegal": {"latex": "\\Omega_\\Lambda"}},
    },
    "de_w": {
        "desc": "Varying constant eq of state",
        "theory": {"camb": None, "classy": {"params": {"Omega_Lambda": 0}}},
        "params": {
            "w": {
                "prior": {"min": -3, "max": -0.333},
                "ref": {"dist": "norm", "loc": -0.99, "scale": 0.02},
                "proposal": 0.02,
                "latex": "w_\\mathrm{DE}",
            }
        },
    },
    "de_w_wa": {
        "desc": "Varying constant eq of state with w(a) = w0 + (1-a) wa",
        "theory": {
            "camb": {"extra_args": {"dark_energy_model": "ppf"}},
            "classy": {"params": {"Omega_Lambda": 0}},
        },
        "params": {
            "w": {
                "prior": {"min": -3, "max": 1},
                "ref": {"dist": "norm", "loc": -0.99, "scale": 0.02},
                "proposal": 0.02,
                "latex": "w_{0,\\mathrm{DE}}",
            },
            "wa": {
                "prior": {"min": -3, "max": 2},
                "ref": {"dist": "norm", "loc": 0, "scale": 0.05},
                "proposal": 0.05,
                "latex": "w_{a,\\mathrm{DE}}",
            },
        },
    },
}

# BBN
bbn_derived_camb: InfoDict = {
    "YpBBN": {"latex": "Y_P^\\mathrm{BBN}"},
    "DHBBN": {"derived": "lambda DH: 10**5*DH", "latex": "10^5 \\mathrm{D}/\\mathrm{H}"},
}
bbn = {
    "consistency": {
        "desc": "Primordial Helium fraction inferred from BBN consistency",
        "theory": {"camb": {"params": bbn_derived_camb}, "classy": None},
        "params": {"yheused": {"latex": "Y_\\mathrm{P}"}},
    },
    "YHe_des_y1": {
        "desc": "Fixed Y_P = 0.245341 (used in DES Y1)",
        "theory": theory,
        "params": {"yhe": 0.245341},
    },
    "YHe": {
        "desc": "Varying primordial Helium fraction",
        "theory": theory,
        "params": {
            "yhe": {
                "prior": {"min": 0.1, "max": 0.5},
                "ref": {"dist": "norm", "loc": 0.237, "scale": 0.006},
                "proposal": 0.006,
                "latex": "Y_\\mathrm{P}",
            }
        },
    },
}

# Reionization
reionization = {
    "std": {
        "desc": "Standard reio, lasting delta_z=0.5",
        "theory": theory,
        "params": {
            "tau": {
                "prior": {"min": 0.01, "max": 0.8},
                "ref": {"dist": "norm", "loc": 0.055, "scale": 0.006},
                "proposal": 0.003,
                "latex": "\\tau_\\mathrm{reio}",
            },
            "zrei": {"latex": "z_\\mathrm{re}"},
        },
    },
    "gauss_prior": {
        "desc": "Standard reio, lasting delta_z=0.5, gaussian prior around tau=0.07",
        "theory": theory,
        "params": {
            "tau": {
                "prior": {"dist": "norm", "loc": 0.07, "scale": 0.02},
                "ref": {"dist": "norm", "loc": 0.07, "scale": 0.01},
                "proposal": 0.005,
                "latex": "\\tau_\\mathrm{reio}",
            },
            "zrei": {"latex": "z_\\mathrm{re}"},
        },
    },
    "irrelevant": {
        "desc": "Irrelevant (NB: only valid for non-CMB or CMB-marged datasets!)",
        "theory": theory,
        "params": {},
    },
}

# EXPERIMENTS ############################################################################

# Precision for calculations without perturbations
base_precision: InfoDict = {
    "camb": {},
    "classy": {},
}

# Precision for CMB analises
cmb_precision = deepcopy(base_precision)
cmb_precision["camb"].update({"lens_potential_accuracy": 1})
cmb_precision["classy"].update({"non linear": "hmcode"})

# Precision for combined CMB + LSS analyses (used for LSS-only too)
cmb_lss_precision = deepcopy(cmb_precision)
cmb_lss_precision["camb"].update({})
cmb_lss_precision["classy"].update({"nonlinear_min_k_max": 20})

default_mcmc_options = {
    "proposal_scale": 1.9,
    "Rminus1_stop": 0.01,
    "Rminus1_cl_stop": 0.2,
}
cmb_sampler_recommended: InfoDict = {
    "mcmc": dict(drag=True, oversample_power=0.4, **default_mcmc_options)
}
cmb_sampler_mcmc: InfoDict = {"mcmc": dict(drag=False, **default_mcmc_options)}

like_cmb: InfoDict = {
    none: {},
    "planck_NPIPE_CamSpec": {
        "desc": "Planck NPIPE CamSpec (native; polarized NPIPE CMB + lensing)",
        "sampler": cmb_sampler_recommended,
        "theory": {
            theo: {"extra_args": cmb_precision[theo]} for theo in ["camb", "classy"]
        },
        "likelihood": {
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
            "planck_NPIPE_highl_CamSpec.TTTEEE": None,
            "planckpr4lensing": {
                "package_install": {
                    "github_repository": "carronj/planck_PR4_lensing",
                    "min_version": "1.0.2",
                }
            },
        },
    },
    "planck_NPIPE_Hillipop": {
        "desc": "Planck NPIPE Hillipop+Lollipop (polarized NPIPE CMB + lensing)",
        "sampler": cmb_sampler_recommended,
        "theory": {
            theo: {"extra_args": cmb_precision[theo]} for theo in ["camb", "classy"]
        },
        "likelihood": {
            "planck_2018_lowl.TT": None,
            "planck_2020_lollipop.lowlE": {
                "package_install": {
                    "pip": "planck-npipe/lollipop",
                    "min_version": "4.1.1",
                }
            },
            "planck_2020_hillipop.TTTEEE": {
                "package_install": {
                    "pip": "planck-npipe/hillipop",
                    "min_version": "4.2.2",
                }
            },
            "planckpr4lensing": {
                "package_install": {
                    "github_repository": "carronj/planck_PR4_lensing",
                    "min_version": "1.0.2",
                }
            },
        },
    },
    "planck_2018": {
        "desc": "Planck 2018 (Polarized CMB + lensing)",
        "sampler": cmb_sampler_recommended,
        "theory": {
            theo: {"extra_args": cmb_precision[theo]} for theo in ["camb", "classy"]
        },
        "likelihood": {
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
            "planck_2018_highl_plik.TTTEEE": None,
            "planck_2018_lensing.clik": None,
        },
    },
    "planck_2018_bk18": {
        "desc": "Planck 2018 (Polarized CMB + lensing) + Bicep/Keck-Array 2018",
        "sampler": cmb_sampler_recommended,
        "theory": {
            theo: {"extra_args": cmb_precision[theo]} for theo in ["camb", "classy"]
        },
        "likelihood": {
            "planck_2018_lowl.TT": None,
            "planck_2018_lowl.EE": None,
            "planck_2018_highl_plik.TTTEEE": None,
            "planck_2018_lensing.clik": None,
            "bicep_keck_2018": None,
        },
    },
    "planck_2018_CMBmarged_lensing": {
        "desc": "Planck 2018 CMB-marginalized lensing only",
        "sampler": cmb_sampler_mcmc,
        "theory": {
            theo: {"extra_args": cmb_precision[theo]} for theo in ["camb", "classy"]
        },
        "likelihood": {"planck_2018_lensing.CMBMarged": None},
    },
}

# Add common CMB derived parameters
derived_params = {
    "sigma8": {"latex": "\\sigma_8"},
    "s8h5": {
        "derived": "lambda sigma8, H0: sigma8*(H0*1e-2)**(-0.5)",
        "latex": "\\sigma_8/h^{0.5}",
    },
    "s8omegamp5": {
        "derived": "lambda sigma8, omegam: sigma8*omegam**0.5",
        "latex": "\\sigma_8 \\Omega_\\mathrm{m}^{0.5}",
    },
    "s8omegamp25": {
        "derived": "lambda sigma8, omegam: sigma8*omegam**0.25",
        "latex": "\\sigma_8 \\Omega_\\mathrm{m}^{0.25}",
    },
    "A": {"derived": "lambda As: 1e9*As", "latex": "10^9 A_\\mathrm{s}"},
    "clamp": {
        "derived": "lambda As, tau: 1e9*As*np.exp(-2*tau)",
        "latex": "10^9 A_\\mathrm{s} e^{-2\\tau}",
    },
    "age": {"latex": "{\\rm{Age}}/\\mathrm{Gyr}"},
    "rdrag": {"latex": "r_\\mathrm{drag}"},
}
for name, m in like_cmb.items():
    # Don't add the derived parameter to the no-CMB case!
    if not m:
        continue
    if "params" not in m:
        m["params"] = dict()
    m["params"].update(derived_params)
    if "cmbmarged" in name.lower():
        m["params"].pop("A")
        m["params"].pop("clamp")
# Some more, in case we want to add them at some point, described in
# https://wiki.cosmos.esa.int/planckpla2015/images/b/b9/Parameter_tag_definitions_2015.pdf
#    "zstar":       {"latex": r"z_*"},
#    "rstar":       {"latex": r"r_*"},
#    "DAstar":      {"latex": r"D_\mathrm{A}/\mathrm{Gpc}"},
#    "zdrag":       {"latex": r"z_\mathrm{drag}"},
#    "kd":          {"latex": r"k_\mathrm{D}"},
#    "thetad":      {"latex": r"100\theta_\mathrm{D}"},
#    "zeq":         {"latex": r"z_\mathrm{eq}"},
#    "keq":         {"latex": r"k_\mathrm{eq}"},
#    "thetaeq":     {"latex": r"100\theta_\mathrm{eq}"},
#    "thetastar":   {"latex": r"100\theta_*"},
#    "thetarseq":   {"latex": r"100\theta_\mathrm{s,eq}"},

like_bao = {
    none: {},
    "BAO_desi_dr2": {
        "desc": "Combined BAO from DESI DR2",
        "theory": theory,
        "likelihood": {"bao.desi_dr2": None},
    },
    "BAO_desi_2024": {
        "desc": "Combined BAO from DESI 2024",
        "theory": theory,
        "likelihood": {"bao.desi_2024_bao_all": None},
    },
    "BAO_planck_2018": {
        "desc": "Baryon acoustic oscillation data from DR12, MGS and 6DF "
        "(Planck 2018 papers)",
        "theory": theory,
        "likelihood": {
            "bao.sixdf_2011_bao": None,
            "bao.sdss_dr7_mgs": None,
            "bao.sdss_dr12_consensus_bao": None,
        },
    },
    "BAO_planck_latest": {
        "desc": "Baryon acoustic oscillation data from BOSS DR12, eBOSS DR16, "
        "MGS and 6DF",
        "theory": theory,
        "likelihood": {
            "bao.sixdf_2011_bao": None,
            "bao.sdss_dr7_mgs": None,
            "bao.sdss_dr16_baoplus_lrg": None,
            "bao.sdss_dr16_baoplus_elg": None,
            "bao.sdss_dr16_baoplus_qso": None,
            "bao.sdss_dr16_baoplus_lyauto": None,
            "bao.sdss_dr16_baoplus_lyxqso": None,
        },
    },
}

like_des: InfoDict = {
    none: {},
    "des_y1_clustering": {
        "desc": "Galaxy clustering from DES Y1",
        "likelihood": {"des_y1.clustering": None},
    },
    "des_y1_galaxy_galaxy": {
        "desc": "Galaxy-galaxy lensing from DES Y1",
        "likelihood": {"des_y1.galaxy_galaxy": None},
    },
    "des_y1_shear": {
        "desc": "Cosmic shear data from DES Y1",
        "likelihood": {"des_y1.shear": None},
    },
    "des_y1_joint": {
        "desc": "Combination of galaxy clustering and weak lensing data from DES Y1",
        "likelihood": {"des_y1.joint": None},
    },
}

for key, value in like_des.items():
    if key is not none:
        value["theory"] = {
            theo: {"extra_args": cmb_lss_precision[theo]} for theo in ["camb", "classy"]
        }
        value["sampler"] = cmb_sampler_recommended

like_sn: InfoDict = {
    none: {},
    "PantheonPlus": {
        "desc": "Supernovae data from the Pantheon+ sample",
        "theory": theory,
        "likelihood": {"sn.pantheonplus": None},
    },
    "Union3": {
        "desc": "Supernovae data from Union3",
        "theory": theory,
        "likelihood": {"sn.union3": None},
    },
    "DESY5": {
        "desc": "Supernovae data from the DES Y5 sample",
        "theory": theory,
        "likelihood": {"sn.desy5": None},
    },
    "DESDovekie": {
        "desc": "Supernovae data from the updated DES-Dovekie Y5 sample",
        "theory": theory,
        "likelihood": {"sn.desdovekie": None},
    },
    "Pantheon": {
        "desc": "Supernovae data from the Pantheon sample",
        "theory": theory,
        "likelihood": {"sn.pantheon": None},
    },
}

like_H0: InfoDict = {
    none: {},
    "Riess2018a": {
        "desc": "Local H0 measurement from Riess et al. 2018a (used in Planck 2018)",
        "theory": theory,
        "likelihood": {"H0.riess2018a": None},
    },
    "Riess201903": {
        "desc": "Local H0 measurement from Riess et al. 2019",
        "theory": theory,
        "likelihood": {"H0.riess201903": None},
    },
    "Riess2020": {
        "desc": "Local H0 measurement from Riess et al. 2020",
        "theory": theory,
        "likelihood": {"H0.riess2020": None},
    },
    "Freedman2020": {
        "desc": "Local H0 measurement from Freedman et al. 2020",
        "theory": theory,
        "likelihood": {"H0.freedman2020": None},
    },
    "Riess2020Mb": {
        "desc": "Local magnitude measurement as from Riess et al. 2020",
        "theory": theory,
        "likelihood": {"H0.riess2020Mb": None, "sn.pantheon": {"use_abs_mag": True}},
    },
}

# SAMPLERS ###############################################################################

sampler: InfoDict = {
    "MCMC": {
        "desc": "MCMC sampler with covmat learning",
        "sampler": {"mcmc": {"covmat": "auto"}},
    },
    "MCMC dragging": {
        "desc": "MCMC fast-dragging sampler with covmat learning",
        "sampler": {
            "mcmc": {
                "drag": True,
                "oversample_power": 0.4,
                "proposal_scale": 1.9,
                "covmat": "auto",
            }
        },
    },
    "PolyChord": {
        "desc": "Nested sampler, affine invariant and multi-modal",
        "sampler": {"polychord": None},
    },
}

# PRESETS ################################################################################

planck_base_model = {
    "primordial": "SFSR",
    "geometry": "flat",
    "hubble": "sound_horizon_last_scattering",
    "matter": "omegab_h2, omegac_h2",
    "neutrinos": "one_heavy_planck",
    "dark_energy": "lambda",
    "bbn": "consistency",
    "reionization": "std",
}
default_sampler = {"sampler": "MCMC dragging"}

preset: InfoDict = dict(
    [
        (none, {"desc": "(No preset chosen)"}),
        # Pure CMB #######################################################
        (
            "planck_NPIPE_CamSpec_camb",
            {
                "desc": "Planck NPIPE CamSpec with CAMB (all native Python)",
                "theory": "camb",
                "like_cmb": "planck_NPIPE_CamSpec",
            },
        ),
        (
            "planck_NPIPE_CamSpec_classy",
            {
                "desc": "Planck NPIPE CamSpec with CLASS (all native Python)",
                "theory": "classy",
                "like_cmb": "planck_NPIPE_CamSpec",
            },
        ),
        (
            "planck_NPIPE_Hillipop_camb",
            {
                "desc": "Planck NPIPE Hillipop+Lollipop with CAMB (all native Python)",
                "theory": "camb",
                "like_cmb": "planck_NPIPE_Hillipop",
            },
        ),
        (
            "planck_NPIPE_Hillipop_classy",
            {
                "desc": "Planck NPIPE Hillipop+Lollipop with CLASS (all native Python)",
                "theory": "classy",
                "like_cmb": "planck_NPIPE_Hillipop",
            },
        ),
        (
            "planck_2018_camb",
            {
                "desc": "Planck 2018 with CAMB",
                "theory": "camb",
                "like_cmb": "planck_2018",
            },
        ),
        (
            "planck_2018_classy",
            {
                "desc": "Planck 2018 with CLASS",
                "theory": "classy",
                "like_cmb": "planck_2018",
            },
        ),
        (
            "planck_2018_bicep_camb",
            {
                "desc": "Planck 2018 + BK18 (with tensor modes) with CAMB",
                "theory": "camb",
                "primordial": "SFSR_t",
                "like_cmb": "planck_2018_bk18",
            },
        ),
        (
            "planck_2018_bicep_classy",
            {
                "desc": "Planck 2018 + BK18 (with tensor modes) with CLASS",
                "theory": "classy",
                "primordial": "SFSR_t",
                "like_cmb": "planck_2018_bk18",
            },
        ),
        # CMB+BAO ######################################################
        (
            "planck_2018_BAO_camb",
            {
                "desc": "Planck 2018 + BAO with CAMB",
                "theory": "camb",
                "like_cmb": "planck_2018",
                "like_bao": "BAO_planck_2018",
            },
        ),
        (
            "planck_2018_BAO_classy",
            {
                "desc": "Planck 2018 + BAO with CLASS",
                "theory": "classy",
                "like_cmb": "planck_2018",
                "like_bao": "BAO_planck_2018",
            },
        ),
        (
            "planck_BAO_latest_camb",
            {
                "desc": "Planck 2018 + eBOSS 16 BAO with CAMB",
                "theory": "camb",
                "like_cmb": "planck_2018",
                "like_bao": "BAO_planck_latest",
            },
        ),
        (
            "planck_BAO_latest_classy",
            {
                "desc": "Planck 2018 + eBOSS 16 BAO with CLASS",
                "theory": "classy",
                "like_cmb": "planck_2018",
                "like_bao": "BAO_planck_latest",
            },
        ),
        # CMB+BAO+SN ###################################################
        (
            "planck_2018_BAO_SN_camb",
            {
                "desc": "Planck 2018 + BAO + SN with CAMB",
                "theory": "camb",
                "like_cmb": "planck_2018",
                "like_bao": "BAO_planck_latest",
                "like_sn": "Pantheon",
            },
        ),
        (
            "planck_2018_BAO_SN_classy",
            {
                "desc": "Planck 2018 + BAO + SN with CLASS",
                "theory": "classy",
                "like_cmb": "planck_2018",
                "like_bao": "BAO_planck_latest",
                "like_sn": "Pantheon",
            },
        ),
        # CMB+DES+BAO+SN ###################################################
        (
            "planck_2018_DES_BAO_SN_camb",
            {
                "desc": "Planck 2018 + DESjoint + BAO + SN with CAMB",
                "theory": "camb",
                "like_cmb": "planck_2018",
                "like_bao": "BAO_planck_latest",
                "like_des": "des_y1_joint",
                "like_sn": "Pantheon",
            },
        ),
        (
            "planck_2018_DES_BAO_SN_classy",
            {
                "desc": "Planck 2018 + DESjoint + BAO + SN with CLASS",
                "theory": "classy",
                "like_cmb": "planck_2018",
                "like_bao": "BAO_planck_latest",
                "like_des": "des_y1_joint",
                "like_sn": "Pantheon",
            },
        ),
    ]
)

# Add planck baseline model
for pre in preset.values():
    pre.update(
        {field: value for field, value in planck_base_model.items() if field not in pre}
    )

# Lensing-only ###################################################
preset.update(
    {
        none: {"desc": "(No preset chosen)"},
        "planck_2018_lensonly_camb": {
            "desc": "Planck 2018 lensing only with CAMB",
            "theory": "camb",
            "like_cmb": "planck_2018_CMBmarged_lensing",
            "like_des": none,
            "primordial": "SFSR_lenspriors",
            "geometry": "flat",
            "hubble": "sound_horizon_lensonly",
            "matter": "omegab_h2_lenspriors",
            "neutrinos": "one_heavy_planck",
            "dark_energy": "lambda",
            "bbn": "consistency",
            "reionization": "irrelevant",
            "sampler": "MCMC",
        },
        "planck_2018_DES_lensingonly_camb": {
            "desc": "Planck 2018 + DES Y1 lensing-only with CAMB",
            "theory": "camb",
            "like_cmb": "planck_2018_CMBmarged_lensing",
            "like_des": "des_y1_shear",
        },
        "planck_2018_DES_lensingonly_classy": {
            "desc": "Planck 2018 + DES Y1 lensing-only with CLASS",
            "theory": "classy",
            "like_cmb": "planck_2018_CMBmarged_lensing",
            "like_des": "des_y1_shear",
        },
    }
)

lensingonly_DES_model = {
    "primordial": "SFSR_DESpriors",
    "geometry": "flat",
    "hubble": "H_DESpriors",
    "matter": "Omegab, Omegam",
    "neutrinos": "one_heavy_planck",
    "dark_energy": "lambda",
    "bbn": "YHe_des_y1",
    "reionization": "irrelevant",
}

# Add planck baseline model
for name, pre in preset.items():
    if "lensingonly" in name:
        pre.update(
            {
                field: value
                for field, value in lensingonly_DES_model.items()
                if field not in pre
            }
        )
    if pre.get("sampler") != "MCMC":
        pre.update(default_sampler)

# BASIC INSTALLATION #####################################################################
install_basic: InfoDict = {
    "theory": theory,
    "likelihood": dict(
        like_cmb["planck_NPIPE_CamSpec"]["likelihood"],
        **{
            # 2018 lensing ensured covmat database also installed
            "planck_2018_lensing.native": None,
            "sn.pantheon": None,
            "bao.sdss_dr12_consensus_final": None,
            "des_y1.joint": None,
        },
    ),
}

install_tests = deepcopy(install_basic)
install_tests["likelihood"].update(
    {
        "planck_2018_highl_plik.TT": None,
        "planck_2018_highl_plik.TT_lite_native": None,
        "planck_2018_highl_CamSpec.TT": None,
        "planck_2018_highl_CamSpec2021.TT": None,
        "bicep_keck_2018": None,
    }
)

# CONTENTS FOR COMBO-BOXED IN A GUI ######################################################

_combo_dict_text = (
    ["Presets", (["preset", "Presets"],)],
    [
        "Cosmological Model",
        (
            ["theory", "Theory code"],
            ["primordial", "Primordial perturbations"],
            ["geometry", "Geometry"],
            ["hubble", "Hubble parameter constraint"],
            ["matter", "Matter sector"],
            ["neutrinos", "Neutrinos and other extra matter"],
            ["dark_energy", "Lambda / Dark energy"],
            ["bbn", "BBN"],
            ["reionization", "Reionization history"],
        ),
    ],
    [
        "Data sets",
        (
            ["like_cmb", "CMB experiments"],
            ["like_bao", "BAO experiments"],
            ["like_des", "DES measurements"],
            ["like_sn", "SN experiments"],
            ["like_H0", "Local H0 measurements"],
        ),
    ],
    ["Sampler", (["sampler", "Samplers"],)],
)
