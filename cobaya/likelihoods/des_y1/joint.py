from cobaya.likelihoods.base_classes import DES


class joint(DES):
    r"""
    Combination of galaxy clustering and weak lensing data from the first year of the
    Dark Energy Survey (DES Y1) \cite{Abbott:2017wau}.
    """

    bibtex_file = "des_y1.bibtex"
