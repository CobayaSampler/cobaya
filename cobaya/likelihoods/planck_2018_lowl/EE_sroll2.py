from .EE import EE


class EE_sroll2(EE):
    """
    Python translation of the Planck sroll2 EE likelihood
    https://web.fe.infn.it/~pagano/low_ell_datasets/sroll2/

    Calibration error is very small compared to EE uncertainty, but calibration can be
    used; otherwise taken to be 1
    """

    install_options = {
        "github_repository": "CobayaSampler/planck_native_data",
        "github_release": "v1",
        "asset": "planck_sroll2_lowE.zip",
        "directory": "planck_sroll2_lowE_native",
    }

    _table_file_name = "sroll2_prob_table.txt"

    @classmethod
    def get_bibtex(cls):
        return cls.get_associated_file_content(".bibtex")
