try:
    from .gui import gui_script
except ImportError:
    # PySide not installed
    # TODO: fix this long logger setup
    from cobaya.log import logger_setup, HandledException
    logger_setup(0, None)
    import logging
    logging.getLogger("cosmo_generator").error(
        "PySide is not installed! "
        "Check Cobaya's documentation for the cosmo_generator "
        "('Basic cosmology runs').")
    raise HandledException
from .autoselect_covmat import get_best_covmat
from .create_input import create_input
from .input_database import planck_base_model, cmb_precision, install_basic
