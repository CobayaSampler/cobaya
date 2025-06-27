import os

from getdist import IniFile, ParamNames
from getdist.parampriors import ParamBounds

from cobaya.typing import InputDict, LikesDict, ParamDict, ParamsDict


def cosmomc_root_to_cobaya_info_dict(root: str, derived_to_input=()) -> InputDict:
    """
    Given the root name of existing cosmomc chain files, tries to construct a Cobaya
    input parameter dictionary with roughly equivalent settings. The output
    dictionary can be used for importance sampling from CosmoMC chains in simple cases
    using Cobaya's 'post'.

    Parameters in the optional derived_to_input list are converted from being derived
    parameters in CosmoMC to non-derived in Cobaya.

    This is by no means guaranteed to produce valid or equivalent results, use at your
    own risk with careful checking! Note that the parameter dictionary will not have
    settings for CAMB, samplers etc, which is OK for importance sampling but you would
    need to add them as necessary to reproduce results.

    Parameter chains in CosmoMC format are available for Planck
    from https://pla.esac.esa.int/pla/#home

    """
    names = ParamNames(root + ".paramnames")
    if os.path.exists(root + ".ranges"):
        ranges = ParamBounds(root + ".ranges")
    else:
        ranges = None
    d: ParamsDict = {}
    info: InputDict = {"params": d}
    for par, name in zip(names.names, names.list()):
        if name.startswith("chi2_") and not name.startswith("chi2__"):
            if name == "chi2_prior":
                continue
            name = name.replace("chi2_", "chi2__")
        if name.startswith("minuslogprior") or name == "chi2":
            continue
        param_dict: ParamDict = {"latex": par.label}
        d[name] = param_dict
        if par.renames:
            param_dict["renames"] = par.renames
        if par.isDerived:
            if name not in derived_to_input:
                param_dict["derived"] = True
            else:
                par.isDerived = False
        if ranges and name in ranges.names:
            if par.isDerived:
                low_up = ranges.getLower(name), ranges.getUpper(name)
                if any(r is not None for r in low_up):
                    param_dict["min"], param_dict["max"] = low_up
            else:
                param_dict["prior"] = [ranges.getLower(name), ranges.getUpper(name)]
    if ranges:
        d.update(ranges.fixedValueDict())
    if names.numberOfName("As") == -1 and names.numberOfName("logA") != -1:
        d["As"] = {"latex": r"A_\mathrm{s}", "value": "lambda logA: 1e-10*np.exp(logA)"}
    if names.numberOfName("cosmomc_theta") == -1 and names.numberOfName("theta") != -1:
        d["cosmomc_theta"] = {
            "latex": r"\theta_{\rm MC}",
            "value": "lambda theta: theta/100",
        }

    # special case for CosmoMC (e.g. Planck) chains
    if os.path.exists(root + ".inputparams"):
        inputs = IniFile(root + ".inputparams")
        for key, value in inputs.params.items():
            if key.startswith("prior["):
                if "prior" not in info:
                    info["prior"] = {}
                param = key[6:-1]
                if param in d:
                    mean, std = (float(v.strip()) for v in value.split())
                    if not names.parWithName(param).isDerived:
                        info["prior"][param + "_prior"] = (
                            "lambda {}: stats.norm.logpdf({}, loc={:g}, scale={:g})".format(
                                param, param, mean, std
                            )
                        )

            if key.startswith("linear_combination["):
                param = key.replace("linear_combination[", "")[:-1]
                prior = inputs.params.get("prior[%s]" % param, None)
                if prior:
                    weights = inputs.params.get(
                        "linear_combination_weights[%s]" % param, None
                    )
                    if not weights:
                        raise ValueError(
                            "linear_combination[%s] prior found but not weights" % param
                        )
                    weights = [float(w.strip()) for w in weights.split()]
                    inputs = value.split()
                    if "prior" not in info:
                        info["prior"] = {}
                    mean, std = (float(v.strip()) for v in prior.split())
                    linear = "".join(f"{_w:+g}*{_p}" for _w, _p in zip(weights, inputs))
                    info["prior"]["SZ"] = (
                        "lambda {}: stats.norm.logpdf({}, loc={:g}, scale={:g})".format(
                            ",".join(inputs), linear, mean, std
                        )
                    )
    if os.path.exists(root + ".likelihoods"):
        info_like: LikesDict = {}
        info["likelihood"] = info_like
        with open(root + ".likelihoods") as f:
            for line in f.readlines():
                if line.strip():
                    like = line.split()[2]
                    info_like[like] = None
    else:
        print(
            'You need to mention in the likelihood block with with "name: None"'
            "for each likelihood in the input chain"
        )

    info["output"] = root
    return info
