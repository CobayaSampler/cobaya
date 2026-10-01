from cobaya.likelihoods.base_classes import CC

# Commit of https://gitlab.com/mmoresco/CCcovariance from which the data are downloaded
_commit = "881413330a7f1e1e5203607d6964db49b4c6c461"


class moresco2020(CC):
    r"""
    Cosmic chronometers :math:`H(z)` measurements of \cite{Moresco:2012},
    \cite{Moresco:2015} and \cite{Moresco:2016} (BC03 stellar population synthesis
    models), with the full covariance matrix of systematics of \cite{Moresco:2020}.
    """

    install_options = {
        "download_url": "https://gitlab.com/mmoresco/CCcovariance/-/archive/"
        f"{_commit}/CCcovariance-{_commit}.tar.gz",
        "data_path": "CCcovariance",
    }
