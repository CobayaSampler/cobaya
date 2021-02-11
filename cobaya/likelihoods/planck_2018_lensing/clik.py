from cobaya.likelihoods._base_classes import _planck_clik_prototype


class clik(_planck_clik_prototype):
    r"""
    Lensing likelihood of Planck's 2018 data release based on temperature+polarization
    map-based lensing reconstruction \cite{Aghanim:2018oex}.
    """
    bibtex_file = 'PlanckLensing2018.bibtex'
