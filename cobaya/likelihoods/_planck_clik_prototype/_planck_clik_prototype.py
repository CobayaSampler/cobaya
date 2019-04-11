"""
.. module:: _planck_clik_prototype

:Synopsis: Definition of the clik-based likelihoods
:Author: Jesus Torrado
         (based on MontePython's version by Julien Lesgourgues and Benjamin Audren)

Family of Planck CMB likelihoods, based on the ``clik`` code.

The Planck 2015 likelihoods defined here are:

- ``planck_2015_lensing``
- ``planck_2015_lensing_cmblikes``
  (a native non-clik, more customizable version of the previous clik-wrapped one)
- ``planck_2015_lowl``
- ``planck_2015_lowTEB``
- ``planck_2015_plikHM_TT``
- ``planck_2015_plikHM_TTTEEE``
- ``planck_2015_plikHM_TTTEEE_unbinned``
- ``planck_2015_plikHM_TT_unbinned``

You can read a description of the different likelihoods in the
`Planck wiki <https://wiki.cosmos.esa.int/planckpla2015/index.php/CMB_spectrum_%26_Likelihood_Code>`_.

.. |br| raw:: html

   <br />

.. note::

   **If you use any of these likelihoods, please cite them as:**
   |br|
   N. Aghanim et al,
   `Planck 2015 results. XI. CMB power spectra, likelihoods, and robustness of parameters`
   `(arXiv:1507.02704) <https://arxiv.org/abs/1507.02704>`_

.. warning::

   ``planck_2015_lowTEB`` cannot be instantiated more than once.


Usage
-----

To use any of the Planck likelihoods, you simply need to mention them in the
``likelihood`` block, or add them using the :doc:`input generator <cosmo_basic_runs>`.

The corresponding nuisance parameters will be added automatically,
so you don't have to care about listing them in the ``params`` block.

The nuisance parameters and their default priors can be found in the ``defaults.yaml``
files in the folder corresponding to each likelihood. They are not reproduced here because
of their length.


Installation
------------

This likelihood can be installed automatically as explained in :doc:`installation_cosmo`.
If are following the instructions there (you should!), you don't need to read the rest
of this section.

.. note::

   By default, the ``gfortran`` compiler will be used, and the ``cfitsio`` library will be
   downloaded and compiled automatically.

   If the installation fails, make sure that the packages ``liblapack3`` and
   ``liblapack-dev`` are installed in the system (in Debian/Ubuntu, simply do
   ``sudo apt install liblapack3 liblapack-dev``).

   If you want to re-compile the Planck likelihood to your liking (e.g. with MKL), simply
   go into the chosen modules installation folder and re-run the ``python waf configure``
   and ``python waf install`` with the desired options.

However, if you wish to install it manually or have a previous installation already in
your system, simply take note of the path to the ``plc-2.0`` and ``plc_2.0`` folders and
mention it below each Planck likelihood as

.. code-block:: yaml

   likelihood:
     planck_2015_lowTEB:
       path: /path/to/planck_2015
     planck_2015_plikHM_TTTEEE:
       path: /path/to/planck_2015


Manual installation of Planck 2015 likelihoods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   For the time being (waiting for Planck 2018's data release), use instead the
   alternative 'clik' code at
   `<https://cdn.cosmologist.info/cosmobox/plc-2.1_py3.tar.bz2>`_, which is
   compatible with Python 3 and gcc `>5`.

Assuming you are installing all your likelihoods under ``/path/to/likelihoods``:

.. code:: bash

   $ cd /path/to/likelihoods
   $ mkdir planck_2015
   $ cd planck_2015
   $ wget https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID=1904
   $ tar xvjf data-action?COSMOLOGY.COSMOLOGY_OID=1904
   $ rm data-action?COSMOLOGY.COSMOLOGY_OID=1904
   $ cd plc-2.0
   $ python waf configure  # [options]

If the last step failed, try adding the option ``--install_all_deps``.
It it doesn't solve it, follow the instructions in the ``readme.md``
file in the ``plc-2.0`` folder.

If you have Intel's compiler and Math Kernel Library (MKL), you may want to also add the
option ``--lapack_mkl=${MKLROOT}`` in the last line to make use of it.

If ``python waf configure`` ended successfully run ``python waf install``
in the same folder. You do **not** need to run ``clik_profile.sh``, as advised.

Now, download the required likelihood files from the
`Planck Legacy Archive <https://pla.esac.esa.int/pla/#cosmology>`_ (Europe) or the
`NASA/IPAC Archive <https://irsa.ipac.caltech.edu/data/Planck/release_2/software/>`_ (US).

For instance, if you want to reproduce the baseline Planck 2015 results,
download the file ``COM_Likelihood_Data-baseline_R2.00.tar.gz``
from any of the two links above, and decompress it under the ``planck_2015`` folder
that you created above.

Finally, download and decompress in the ``planck_2015`` folder the last release at
`this repo <https://github.com/CobayaSampler/planck_supp_data_and_covmats/releases>`_.

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Global
import os
import sys
import numpy as np
import logging
import six

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException
from cobaya.conventions import _path_install, _likelihood
from cobaya.input import get_default_info
from cobaya.install import pip_install, download_file


class _planck_clik_prototype(Likelihood):

    def initialize(self):
        if self.path:
            has_clik = False
        else:
            try:
                import clik
                has_clik = True
            except ImportError:
                has_clik = False
        if not has_clik:
            if not self.path:
                if self.path_install:
                    self.path_clik = os.path.join(
                        self.path_install, "code", common_path)
                else:
                    self.log.error("No path given to the Planck likelihood. Set the "
                                   "likelihood property 'path' or the common property "
                                   "'%s'.", _path_install)
                    raise HandledException
            else:
                self.path_clik = self.path
            self.log.info("Importing clik from %s", self.path_clik)
            # test and import clik
            is_installed_clik(self.path_clik, log_and_fail=True, import_it=False)
            import clik
        # Loading the likelihood data
        if not os.path.isabs(self.clik_file):
            self.path_data = getattr(self, "path_data", os.path.join(
                self.path or self.path_install, "data", common_path))
            self.clik_file = os.path.join(self.path_data, self.clik_file)
        # for lensing, some routines change. Initializing a flag for easier
        # testing of this condition
        self.lensing = "lensing" in self.name
        try:
            self.clik = (
                clik.clik_lensing(self.clik_file) if self.lensing else clik.clik(self.clik_file))
        except clik.lkl.CError:
            self.log.error(
                "The .clik file was not found where specified in the 'clik_file' field "
                "of the settings of this likelihood. Maybe the 'path' given is not "
                "correct? The full path where the .clik file was searched for is '%s'",
                self.clik_file)
            raise HandledException
        self.expected_params = list(self.clik.extra_parameter_names)
        # py3 lensing bug
        if "b'A_planck'" in self.expected_params:
            self.expected_params[self.expected_params.index("b'A_planck'")] = "A_planck"
        # line added to deal with a bug in planck likelihood release:
        # A_planck called A_Planck in plik_lite
        if "plikHM_lite" in self.name:
            i = self.expected_params.index('A_Planck')
            self.expected_params[i] = 'A_planck'
            self.log.info("Corrected nuisance parameter name A_Planck to A_planck")
        # Check that the parameters are the right ones
        assert set(self.input_params) == set(self.expected_params), (
                "Likelihoods parameters do not coincide with the ones clik understands. Have: %s, Expected: %s"
                % (set(self.input_params), set(self.expected_params)))
        # Placeholder for vector passed to clik
        self.l_maxs = self.clik.get_lmax()
        length = (len(self.l_maxs) if self.lensing else len(self.clik.get_has_cl()))
        self.vector = np.zeros(np.sum(self.l_maxs) + length + len(self.expected_params))

    def add_theory(self):
        # State requisites to the theory code
        requested_cls = ["tt", "ee", "bb", "te", "tb", "eb"]
        if self.lensing:
            has_cl = [lmax != -1 for lmax in self.l_maxs]
            requested_cls = ["pp"] + requested_cls
        else:
            has_cl = self.clik.get_has_cl()
        self.requested_cls = [cl for cl, i in zip(requested_cls, has_cl) if int(i)]
        self.l_maxs_cls = [lmax for lmax, i in zip(self.l_maxs, has_cl) if int(i)]
        self.theory.needs(Cl=dict(zip(self.requested_cls, self.l_maxs_cls)))

    def logp(self, **params_values):
        # get Cl's from the theory code
        cl = self.theory.get_cl()
        # fill with Cl's
        self.vector[:-len(self.expected_params)] = np.concatenate(
            [(cl[spectrum][:1 + lmax] if spectrum not in ["tb", "eb"]
              else np.zeros(1 + lmax))
             for spectrum, lmax in zip(self.requested_cls, self.l_maxs_cls)])
        # fill with likelihood parameters
        self.vector[-len(self.expected_params):] = (
            [params_values[p] for p in self.expected_params])
        loglike = self.clik(self.vector)[0]
        # "zero" of clik
        if np.allclose(loglike, -1e30):
            loglike = -np.inf
        return loglike

    def close(self):
        del self.clik  # MANDATORY: forces deallocation of the Cython class
        # Actually, it does not work for low-l likelihoods, which is quite dangerous!


# Installation routines ##################################################################

# path to be shared by all Planck likelihoods
common_path = "planck_2015"

# To see full clik build output even if installs OK (e.g. to check warnings)
_clik_verbose = any(
    [(s in os.getenv('TRAVIS_COMMIT_MESSAGE', '')) for s in ["clik", "planck"]])
# Don't try again to install clik if it failed for a previous likelihood
_clik_install_failed = False


def is_installed_clik(path, log_and_fail=False, import_it=True):
    log = logging.getLogger("clik")
    if os.path.exists(path) and len(os.listdir(path)):
        clik_path = os.path.join(path, os.listdir(path)[0], 'lib/python/site-packages')
    else:
        clik_path = None
    if not clik_path or not os.path.exists(clik_path):
        if log_and_fail:
            log.error("The given folder does not exist: '%s'", clik_path or path)
            raise HandledException
        return False
    sys.path.insert(0, clik_path)
    try:
        if import_it:
            import clik
        return True
    except:
        print('Failed to import clik')
        if log_and_fail:
            log.error("Error importing click from: '%s'", clik_path)
            raise HandledException
        return False


def execute(command):
    from subprocess import Popen, PIPE, STDOUT
    if _clik_verbose:
        process = Popen(command, stdout=PIPE, stderr=STDOUT)
        out = []
        while True:
            nextline = process.stdout.readline()
            if nextline == b"" and process.poll() is not None:
                break
            if six.PY3:
                sys.stdout.buffer.write(nextline)
            else:
                sys.stdout.write(nextline)
            out.append(nextline)
            sys.stdout.flush()
        _, err = process.communicate()
        return b"finished successfully" in out[-1]
    else:
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        out, err = process.communicate()
        OK = b"finished successfully" in out.split(b"\n")[-2]
        if not OK:
            print(out.decode('utf-8'))
            print(err.decode('utf-8'))
        return OK


def install_clik(path, no_progress_bars=False):
    log = logging.getLogger("clik")
    log.info("Installing pre-requisites...")
    for req in ("cython", "astropy"):
        exit_status = pip_install(req)
        if exit_status:
            log.error("Failed installing '%s'.", req)
            raise HandledException
    log.info("Downloading...")
    click_url = 'https://cdn.cosmologist.info/cosmobox/plc-2.1_py3.tar.bz2'
    if not download_file(click_url, path, decompress=True,
                         no_progress_bars=no_progress_bars, logger=log):
        log.error("Not possible to download clik.")
        return False
    source_dir = os.path.join(path, os.listdir(path)[0])
    log.info('Installing from directory %s' % source_dir)
    if True:  # should be fixed
        log.info("Patching origin of cfitsio")
        cfitsio_filename = os.path.join(source_dir, "waf_tools", "cfitsio.py")
        with open(cfitsio_filename, "r") as cfitsio_file:
            lines = cfitsio_file.readlines()
            i_offending = next(i for i, l in enumerate(lines) if ".tar.gz" in l)
            lines[i_offending] = (
                "  atl.installsmthg_pre(ctx,"
                "'http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3280.tar.gz',"
                "'cfitsio3280.tar.gz')\n")
        with open(cfitsio_filename, "w") as cfitsio_file:
            cfitsio_file.write("".join(lines))
    cwd = os.getcwd()
    try:
        os.chdir(source_dir)
        log.info("Configuring... (and maybe installing dependencies...)")
        if not execute([sys.executable, "waf", "configure", "--install_all_deps"]):
            log.error("Configuration failed!")
            return False

        log.info("Compiling...")
        if not execute([sys.executable, "waf", "install"]):
            log.error("Compilation failed!")
            return False
    finally:
        os.chdir(cwd)
    log.info("Finished!")
    return True


def get_product_id_and_clik_file(name):
    """Gets the PLA product info from the defaults file."""
    defaults = get_default_info(name, _likelihood)[_likelihood][name]
    return defaults.get("product_id"), defaults.get("clik_file")


def is_installed(**kwargs):
    result = True
    if kwargs["code"]:
        result &= is_installed_clik(os.path.realpath(
            os.path.join(kwargs["path"], "code", common_path)))
    if kwargs["data"]:
        _, filename = get_product_id_and_clik_file(kwargs["name"])
        result &= os.path.exists(os.path.realpath(
            os.path.join(kwargs["path"], "data", common_path, filename)))
        from cobaya.likelihoods.planck_2015_lensing_cmblikes import \
            is_installed as is_installed_supp
        result &= is_installed_supp(**kwargs)
    return result


def install(path=None, name=None, force=False, code=True, data=True,
            no_progress_bars=False):
    log = logging.getLogger(name)
    import platform
    if platform.system() == "Windows":
        log.error("Not compatible with Windows.")
        return False
    global _clik_install_failed
    if _clik_install_failed:
        log.info("Previous clik install failed, skipping")
        return False
    # Create common folders: all planck likelihoods share install folder for code and data
    paths = {}
    for s in ("code", "data"):
        if eval(s):
            paths[s] = os.path.realpath(os.path.join(path, s, common_path))
            if not os.path.exists(paths[s]):
                os.makedirs(paths[s])
    success = True
    # Install clik
    if code and (not is_installed_clik(paths["code"]) or force):
        log.info("Installing the clik code.")
        success *= install_clik(paths["code"], no_progress_bars=no_progress_bars)
        if not success:
            log.warning("clik code installation failed! "
                        "Try configuring+compiling by hand at " + paths["code"])
        _clik_install_failed = True
    if data:
        # 2nd test, in case the code wasn't there but the data is:
        if force or not is_installed(path=path, name=name, code=False, data=True):
            # Extract product_id
            product_id, _ = get_product_id_and_clik_file(name)
            # Download and decompress the particular likelihood
            log.info("Downloading likelihood data...")
            prefix = (r"https://pla.esac.esa.int/pla-sl/"
                      "data-action?COSMOLOGY.COSMOLOGY_OID=")
            if not download_file(prefix + product_id, paths["data"], decompress=True,
                                 logger=log, no_progress_bars=no_progress_bars):
                log.error("Not possible to download this likelihood.")
                success = False
            # Additional data and covmats
            from cobaya.likelihoods.planck_2015_lensing_cmblikes import \
                install as install_supp
            success *= install_supp(path=path, force=force, code=code, data=data,
                                    no_progress_bars=no_progress_bars)
    return success
