r"""
.. module:: _planck_clik_prototype

:Synopsis: Definition of the clik-based likelihoods
:Author: Jesus Torrado
         (based on MontePython's version by Julien Lesgourgues and Benjamin Audren)

Family of Planck CMB likelihoods, as interfaces to the official ``clik`` code (and some
native ``cmblikes`` ones) You can find a description of the different likelihoods in the
`Planck wiki <https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code>`_.

.. |br| raw:: html

   <br />

.. note::

   **If you use any of these likelihoods, please cite them as:**
   |br|
   N. Aghanim et al,
   `Planck 2015 results. XI. CMB power spectra, likelihoods, and robustness of parameters`
   `(arXiv:1507.02704) <https://arxiv.org/abs/1507.02704>`_

The Planck 2018 likelihoods defined here are: (*new in 2.0*)

- ``planck_2018_lowl``: low-:math:`\ell` temperature
- ``planck_2018_lowE``: low-:math:`\ell` E polarization
- ``planck_2018_plikHM_TT``: high-:math:`\ell` temperature
- ``planck_2018_plikHM_TTTEEE``:
  high-:math:`\ell` temperature, polarization and cross-correlated
- ``planck_2018_plikHM_TT_unbinned``, ``planck_2018_plikHM_TTTEEE_unbinned``:
  unbinned-in-:math:`\ell` versions of the baseline ones
- ``planck_2018_plikHM_TT_lite``, ``planck_2018_plikHM_TTTEEE_lite``:
  foreground-marginalized versions of the baseline ones
- ``planck_2015_clik_lensing``: lensing temperature+polarisation-based,
  based on the official ``clik`` code.
- ``planck_2015_cmblikes_lensing``: native version of the ``clik``-based one above,
  more customizable.
- ``planck_2015_cmblikes_lensing_cmbmarged``: CMB-marginalized,
  temperature+polarisation-based lensing likelihood.
  Do not combine with any of the ones above!

The Planck 2015 likelihoods defined here are:

- ``planck_2015_lowl``
- ``planck_2015_lowTEB``
- ``planck_2015_plikHM_TT``
- ``planck_2015_plikHM_TT_unbinned``
- ``planck_2015_plikHM_TTTEEE``
- ``planck_2015_plikHM_TTTEEE_unbinned``
- ``planck_2015_lensing``
- ``planck_2015_lensing_cmblikes``
  (a native non-clik, more customizable version of the previous clik-wrapped one)

.. warning::

   The Planck 2015 likelihoods have been superseeded by the 2018 release, and will
   eventually be deprecated.

.. warning::

   ``planck_2015_lowTEB`` cannot be instantiated more than once. This should have no
   consequence when calling ``cobaya-run`` from the shell, but will impede running
   a sampler or defining a model more than once per Python interpreter session.


Usage
-----

To use any of the Planck likelihoods, you simply need to mention them in the
``likelihood`` block, or add them using the :doc:`input generator <cosmo_basic_runs>`.

The corresponding nuisance parameters will be added automatically,
so you don't have to care about listing them in the ``params`` block.

The nuisance parameters and their default priors can be found in the
``[likelihood_name].yaml`` files in the folder corresponding to each likelihood.
They are not reproduced here because of their length.


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
import textwrap
import six

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from cobaya.conventions import _path_install, _likelihood, _params, _prior
from cobaya.input import get_default_info, HasDefaults
from cobaya.install import pip_install, download_file
from cobaya.tools import are_different_params_lists, create_banner
from cobaya.yaml import yaml_load


_deprecation_msg_2015 = create_banner("""
The likelihoods from the Planck 2015 data release have been superseeded
by the 2018 ones, and will eventually be deprecated.
""")


class _planck_clik_prototype(Likelihood, HasDefaults):

    def initialize(self):
        if "2015" in self.name:
            for line in _deprecation_msg_2015.split("\n"):
                self.log.warning(line)
        code_path = common_path
        data_path = get_data_path(self.name)
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
                    self.path_clik = os.path.join(self.path_install, "code", code_path)
                else:
                    raise LoggedError(
                        self.log, "No path given to the Planck likelihood. Set the "
                        "likelihood property 'path' or the common property "
                        "'%s'.", _path_install)
            else:
                self.path_clik = self.path
            self.log.info("Importing clik from %s", self.path_clik)
            # test and import clik
            is_installed_clik(self.path_clik, log_and_fail=True, import_it=False)
            import clik
        # Loading the likelihood data
        if not os.path.isabs(self.clik_file):
            self.path_data = getattr(self, "path_data", os.path.join(
                self.path or self.path_install, "data", data_path))
            self.clik_file = os.path.join(self.path_data, self.clik_file)
        # Differences in the wrapper for lensing and non-lensing likes
        self.lensing = clik.try_lensing(self.clik_file)
        try:
            self.clik = (
                clik.clik_lensing(self.clik_file) if self.lensing else clik.clik(self.clik_file))
        except clik.lkl.CError:
            # Is it that the file was not found?
            if not os.path.exists(self.clik_file):
                raise LoggedError(
                    self.log, "The .clik file was not found where specified in the "
                    "'clik_file' field of the settings of this likelihood. Maybe the "
                    "'path' given is not correct? The full path where the .clik file was "
                    "searched for is '%s'", self.clik_file)
            # Else: unknown clik error
            self.log.error("An unexpected error occurred in clik (possibly related to "
                           "multiple simultaneous initialization, or simultaneous "
                           "initialization of incompatible likelihoods (e.g. polarised "
                           "vs non-polarised 'lite' likelihoods. See error info below:")
            raise
        self.expected_params = list(self.clik.extra_parameter_names)
        # py3 lensing bug
        if "b'A_planck'" in self.expected_params:
            self.expected_params[self.expected_params.index("b'A_planck'")] = "A_planck"
        # line added to deal with a bug in planck likelihood release:
        # A_planck called A_Planck in plik_lite
        if "_lite" in self.name:
            i = self.expected_params.index('A_Planck')
            self.expected_params[i] = 'A_planck'
        # Check that the parameters are the right ones
        differences = are_different_params_lists(
            self.input_params, self.expected_params, name_A="given", name_B="expected")
        if differences:
            raise LoggedError(
                self.log, "Configuration error in parameters: %r. "
                "If this has happened without you fiddling with the defaults, "
                "please open an issue in GitHub.", differences)
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
        cl = self.theory.get_Cl()
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

    @classmethod
    def get_defaults(cls, return_yaml=False):
        defaults = textwrap.dedent(cls.defaults)
        nuisances = yaml_load(defaults)[_likelihood][cls.__name__].get("nuisance", [])
        release = next(year for year in ["2015", "2018"] if year in cls.__name__)
        if nuisances:
            defaults += "\n\n%s:" % _params
        for nui in nuisances:
            defaults += nuisance_params[release][nui]
        if any([nuisance_priors[release].get(nui) for nui in nuisances]):
            defaults += "\n\n%s:" % _prior
        for nui in nuisances:
            defaults += nuisance_priors[release].get(nui, "")
        defaults = defaults.strip()
        if return_yaml:
            return defaults
        else:
            return yaml_load(defaults)

# Installation routines ##################################################################

# path to be shared by all Planck likelihoods
common_path = "planck"

# To see full clik build output even if installs OK (e.g. to check warnings)
_clik_verbose = any(
    [(s in os.getenv('TRAVIS_COMMIT_MESSAGE', '')) for s in ["clik", "planck"]])
# Don't try again to install clik if it failed for a previous likelihood
try:
    _clik_install_failed
except NameError:
    _clik_install_failed = False


def get_data_path(name):
    return common_path + "_%s" % get_release(name)


def get_release(name):
    return next(re for re in ["2015", "2018"] if re in name)


def is_installed_clik(path, log_and_fail=False, import_it=True):
    log = logging.getLogger("clik")
    if os.path.exists(path) and len(os.listdir(path)):
        clik_path = os.path.join(path, os.listdir(path)[0], 'lib/python/site-packages')
    else:
        clik_path = None
    if not clik_path or not os.path.exists(clik_path):
        if log_and_fail:
            raise LoggedError(log, "The given folder does not exist: '%s'", clik_path or path)
        return False
    sys.path.insert(0, clik_path)
    try:
        if import_it:
            import clik
        return True
    except:
        print('Failed to import clik')
        if log_and_fail:
            raise LoggedError(log, "Error importing click from: '%s'", clik_path)
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
            raise LoggedError(log, "Failed installing '%s'.", req)
    log.info("Downloading...")
    click_url = 'https://cdn.cosmologist.info/cosmobox/plc-2.1_py3.tar.bz2'
    if not download_file(click_url, path, decompress=True,
                         no_progress_bars=no_progress_bars, logger=log):
        log.error("Not possible to download clik.")
        return False
    source_dir = os.path.join(path, os.listdir(path)[0])
    log.info('Installing from directory %s' % source_dir)
    # The following code patches a problem with the download source of cfitsio.
    # Left here in case the FTP server breaks again.
    # if True:  # should be fixed: maybe a ping to the FTP server???
    #     log.info("Patching origin of cfitsio")
    #     cfitsio_filename = os.path.join(source_dir, "waf_tools", "cfitsio.py")
    #     with open(cfitsio_filename, "r") as cfitsio_file:
    #         lines = cfitsio_file.readlines()
    #         i_offending = next(i for i, l in enumerate(lines) if ".tar.gz" in l)
    #         lines[i_offending] = (
    #             "  atl.installsmthg_pre(ctx,"
    #             "'http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3280.tar.gz',"
    #             "'cfitsio3280.tar.gz')\n")
    #     with open(cfitsio_filename, "w") as cfitsio_file:
    #         cfitsio_file.write("".join(lines))
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
    code_path = common_path
    data_path = get_data_path(kwargs["name"])
    result = True
    if kwargs["code"]:
        result &= is_installed_clik(os.path.realpath(
            os.path.join(kwargs["path"], "code", code_path)))
    if kwargs["data"]:
        _, filename = get_product_id_and_clik_file(kwargs["name"])
        result &= os.path.exists(os.path.realpath(
            os.path.join(kwargs["path"], "data", data_path, filename)))
        from cobaya.likelihoods.planck_2015_lensing_cmblikes import \
            is_installed as is_installed_supp
        result &= is_installed_supp(**kwargs)
    return result


def install(path=None, name=None, force=False, code=True, data=True,
            no_progress_bars=False):
    log = logging.getLogger(name)
    path_names = {"code": common_path, "data": get_data_path(name)}
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
            paths[s] = os.path.realpath(os.path.join(path, s, path_names[s]))
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
            if "2015" in name:
                # Extract product_id
                product_id, _ = get_product_id_and_clik_file(name)
                # Download and decompress the particular likelihood
                url = (r"https://pla.esac.esa.int/pla-sl/"
                       "data-action?COSMOLOGY.COSMOLOGY_OID=" + product_id)
            else:
                # OVERRIDE! -- for baseline only
                url = 'http://localhost:8000//home/jesus/scratch/plc_3.0_release/baseline.tar.gz'
            log.info("Downloading likelihood data...")
            if not download_file(url, paths["data"], decompress=True,
                                 logger=log, no_progress_bars=no_progress_bars):
                log.error("Not possible to download this likelihood.")
                success = False
            # Additional data and covmats
            from cobaya.likelihoods.planck_2015_lensing_cmblikes import \
                install as install_supp
            success *= install_supp(path=path, force=force, code=code, data=data,
                                    no_progress_bars=no_progress_bars)
    return success


# Default parameters subclasses ##########################################################

params_2015_calib = r"""
  A_planck:
    prior:
      dist: norm
      loc: 1
      scale: 0.0025
    ref:
      dist: norm
      loc: 1
      scale: 0.002
    proposal: 0.0005
    latex: y_\mathrm{cal}
    renames: calPlanck"""

params_2015_TT = r"""
  cib_index: -1.3
  A_cib_217:
    prior:
      dist: uniform
      min: 0
      max: 200
    ref:
      dist: norm
      loc: 65
      scale: 10
    proposal: 1.2
    latex: A^\mathrm{CIB}_{217}
    renames: acib217
  xi_sz_cib:
    prior:
      dist: uniform
      min: 0
      max: 1
    ref:
      dist: halfnorm
      loc: 0
      scale: 0.1
    proposal: 0.1
    latex: \xi^{\mathrm{tSZ}\times\mathrm{CIB}}
    renames: xi
  A_sz:
    prior:
      dist: uniform
      min: 0
      max: 10
    ref:
      dist: norm
      loc: 5
      scale: 2
    proposal: 0.6
    latex: A^\mathrm{tSZ}_{143}
    renames: asz143
  ps_A_100_100:
    prior:
      dist: uniform
      min: 0
      max: 400
    ref:
      dist: norm
      loc: 250
      scale: 24
    proposal: 17
    latex: A^\mathrm{PS}_{100}
    renames: aps100
  ps_A_143_143:
    prior:
      dist: uniform
      min: 0
      max: 400
    ref:
      dist: norm
      loc: 40
      scale: 10
    proposal: 3
    latex: A^\mathrm{PS}_{143}
    renames: aps143
  ps_A_143_217:
    prior:
      dist: uniform
      min: 0
      max: 400
    ref:
      dist: norm
      loc: 40
      scale: 12
    proposal: 2
    latex: A^\mathrm{PS}_{\mathrm{143}\times\mathrm{217}}
    renames: aps143217
  ps_A_217_217:
    prior:
      dist: uniform
      min: 0
      max: 400
    ref:
      dist: norm
      loc: 100
      scale: 13
    proposal: 2.5
    latex: A^\mathrm{PS}_{217}
    renames: aps217
  ksz_norm:
    prior:
      dist: uniform
      min: 0
      max: 10
    ref:
      dist: halfnorm
      loc: 0
      scale: 3
    proposal: 1
    latex: A^\mathrm{kSZ}
    renames: aksz
  gal545_A_100:
    prior:
      dist: norm
      loc: 7
      scale: 2
    ref:
      dist: norm
      loc: 7
      scale: 2
    proposal: 1
    latex: A^\mathrm{dustTT}_{100}
    renames: kgal100
  gal545_A_143:
    prior:
      dist: norm
      loc: 9
      scale: 2
    ref:
      dist: norm
      loc: 9
      scale: 2
    proposal: 1
    latex: A^\mathrm{dustTT}_{143}
    renames: kgal143
  gal545_A_143_217:
    prior:
      dist: norm
      loc: 21
      scale: 8.5
    ref:
      dist: norm
      loc: 21
      scale: 4
    proposal: 1.5
    latex: A^\mathrm{dustTT}_{\mathrm{143}\times\mathrm{217}}
    renames: kgal143217
  gal545_A_217:
    prior:
      dist: norm
      loc: 80
      scale: 20
    ref:
      dist: norm
      loc: 80
      scale: 15
    proposal: 2
    latex: A^\mathrm{dustTT}_{217}
    renames: kgal217
  calib_100T:
    prior:
      dist: norm
      loc: 0.999
      scale: 0.001
    ref:
      dist: norm
      loc: 0.999
      scale: 0.001
    proposal: 0.0005
    latex: c_{100}
    renames: cal0
  calib_217T:
    prior:
      dist: norm
      loc: 0.995
      scale: 0.002
    ref:
      dist: norm
      loc: 0.995
      scale: 0.002
    proposal: 0.001
    latex: c_{217}
    renames: cal2"""

params_2015_TEEE = r"""
  A_pol: 1
  calib_100P: 1
  calib_143P: 1
  calib_217P: 1
  galf_EE_index: -2.4
  galf_TE_index: -2.4
  bleak_epsilon_0_0T_0E: 0
  bleak_epsilon_1_0T_0E: 0
  bleak_epsilon_2_0T_0E: 0
  bleak_epsilon_3_0T_0E: 0
  bleak_epsilon_4_0T_0E: 0
  bleak_epsilon_0_0T_1E: 0
  bleak_epsilon_1_0T_1E: 0
  bleak_epsilon_2_0T_1E: 0
  bleak_epsilon_3_0T_1E: 0
  bleak_epsilon_4_0T_1E: 0
  bleak_epsilon_0_0T_2E: 0
  bleak_epsilon_1_0T_2E: 0
  bleak_epsilon_2_0T_2E: 0
  bleak_epsilon_3_0T_2E: 0
  bleak_epsilon_4_0T_2E: 0
  bleak_epsilon_0_1T_1E: 0
  bleak_epsilon_1_1T_1E: 0
  bleak_epsilon_2_1T_1E: 0
  bleak_epsilon_3_1T_1E: 0
  bleak_epsilon_4_1T_1E: 0
  bleak_epsilon_0_1T_2E: 0
  bleak_epsilon_1_1T_2E: 0
  bleak_epsilon_2_1T_2E: 0
  bleak_epsilon_3_1T_2E: 0
  bleak_epsilon_4_1T_2E: 0
  bleak_epsilon_0_2T_2E: 0
  bleak_epsilon_1_2T_2E: 0
  bleak_epsilon_2_2T_2E: 0
  bleak_epsilon_3_2T_2E: 0
  bleak_epsilon_4_2T_2E: 0
  bleak_epsilon_0_0E_0E: 0
  bleak_epsilon_1_0E_0E: 0
  bleak_epsilon_2_0E_0E: 0
  bleak_epsilon_3_0E_0E: 0
  bleak_epsilon_4_0E_0E: 0
  bleak_epsilon_0_0E_1E: 0
  bleak_epsilon_1_0E_1E: 0
  bleak_epsilon_2_0E_1E: 0
  bleak_epsilon_3_0E_1E: 0
  bleak_epsilon_4_0E_1E: 0
  bleak_epsilon_0_0E_2E: 0
  bleak_epsilon_1_0E_2E: 0
  bleak_epsilon_2_0E_2E: 0
  bleak_epsilon_3_0E_2E: 0
  bleak_epsilon_4_0E_2E: 0
  bleak_epsilon_0_1E_1E: 0
  bleak_epsilon_1_1E_1E: 0
  bleak_epsilon_2_1E_1E: 0
  bleak_epsilon_3_1E_1E: 0
  bleak_epsilon_4_1E_1E: 0
  bleak_epsilon_0_1E_2E: 0
  bleak_epsilon_1_1E_2E: 0
  bleak_epsilon_2_1E_2E: 0
  bleak_epsilon_3_1E_2E: 0
  bleak_epsilon_4_1E_2E: 0
  bleak_epsilon_0_2E_2E: 0
  bleak_epsilon_1_2E_2E: 0
  bleak_epsilon_2_2E_2E: 0
  bleak_epsilon_3_2E_2E: 0
  bleak_epsilon_4_2E_2E: 0
  galf_EE_A_100:
    prior:
      dist: norm
      loc: 0.060
      scale: 0.012
    ref:
      dist: norm
      loc: 0.06
      scale: 0.012
    proposal: 0.006
    latex: A^\mathrm{dustEE}_{100}
    renames: galfEE100
  galf_EE_A_100_143:
    prior:
      dist: norm
      loc: 0.050
      scale: 0.015
    ref:
      dist: norm
      loc: 0.05
      scale: 0.015
    proposal: 0.007
    latex: A^\mathrm{dustEE}_{\mathrm{100}\times\mathrm{143}}
    renames: galfEE100143
  galf_EE_A_100_217:
    prior:
      dist: norm
      loc: 0.110
      scale: 0.033
    ref:
      dist: norm
      loc: 0.11
      scale: 0.03
    proposal: 0.02
    latex: A^\mathrm{dustEE}_{\mathrm{100}\times\mathrm{217}}
    renames: galfEE100217
  galf_EE_A_143:
    prior:
      dist: norm
      loc: 0.10
      scale: 0.02
    ref:
      dist: norm
      loc: 0.1
      scale: 0.02
    proposal: 0.01
    latex: A^\mathrm{dustEE}_{143}
    renames: galfEE143
  galf_EE_A_143_217:
    prior:
      dist: norm
      loc: 0.240
      scale: 0.048
    ref:
      dist: norm
      loc: 0.24
      scale: 0.05
    proposal: 0.03
    latex: A^\mathrm{dustEE}_{\mathrm{143}\times\mathrm{217}}
    renames: galfEE143217
  galf_EE_A_217:
    prior:
      dist: norm
      loc: 0.72
      scale: 0.14
    ref:
      dist: norm
      loc: 0.72
      scale: 0.15
    proposal: 0.07
    latex: A^\mathrm{dustEE}_{217}
    renames: galfEE217
  galf_TE_A_100:
    prior:
      dist: norm
      loc: 0.140
      scale: 0.042
    ref:
      dist: norm
      loc: 0.14
      scale: 0.04
    proposal: 0.02
    latex: A^\mathrm{dustTE}_{100}
    renames: galfTE100
  galf_TE_A_100_143:
    prior:
      dist: norm
      loc: 0.120
      scale: 0.036
    ref:
      dist: norm
      loc: 0.12
      scale: 0.04
    proposal: 0.02
    latex: A^\mathrm{dustTE}_{\mathrm{100}\times\mathrm{143}}
    renames: galfTE100143
  galf_TE_A_100_217:
    prior:
      dist: norm
      loc: 0.30
      scale: 0.09
    ref:
      dist: norm
      loc: 0.3
      scale: 0.09
    proposal: 0.05
    latex: A^\mathrm{dustTE}_{\mathrm{100}\times\mathrm{217}}
    renames: galfTE100217
  galf_TE_A_143:
    prior:
      dist: norm
      loc: 0.240
      scale: 0.072
    ref:
      dist: norm
      loc: 0.24
      scale: 0.07
    proposal: 0.04
    latex: A^\mathrm{dustTE}_{143}
    renames: galfTE143
  galf_TE_A_143_217:
    prior:
      dist: norm
      loc: 0.60
      scale: 0.18
    ref:
      dist: norm
      loc: 0.60
      scale: 0.18
    proposal: 0.1
    latex: A^\mathrm{dustTE}_{\mathrm{143}\times\mathrm{217}}
    renames: galfTE143217
  galf_TE_A_217:
    prior:
      dist: norm
      loc: 1.80
      scale: 0.54
    ref:
      dist: norm
      loc: 1.8
      scale: 0.5
    proposal: 0.3
    latex: A^\mathrm{dustTE}_{217}
    renames: galfTE217"""

params_2018_calib = r"""
  A_planck:
    prior:
      dist: norm
      loc: 1
      scale: 0.0025
    ref:
      dist: norm
      loc: 1
      scale: 0.002
    proposal: 0.0005
    latex: y_\mathrm{cal}
    renames: calPlanck"""

params_2018_TT = r"""
  # CIB & SZ
  cib_index: -1.3
  A_cib_217:
    prior:
      dist: uniform
      min: 0
      max: 200
    ref:
      dist: norm
      loc: 67
      scale: 10
    proposal: 1.2
    latex: A^\mathrm{CIB}_{217}
    renames: acib217
  xi_sz_cib:
    prior:
      dist: uniform
      min: 0
      max: 1
    ref:
      dist: halfnorm
      loc: 0
      scale: 0.1
    proposal: 0.1
    latex: \xi^{\mathrm{tSZ}\times\mathrm{CIB}}
    renames: xi
  A_sz:
    prior:
      dist: uniform
      min: 0
      max: 10
    ref:
      dist: norm
      loc: 7
      scale: 2
    proposal: 0.6
    latex: A^\mathrm{tSZ}_{143}
    renames: asz143
  ksz_norm:
    prior:
      dist: uniform
      min: 0
      max: 10
    ref:
      dist: halfnorm
      loc: 0
      scale: 3
    proposal: 1
    latex: A^\mathrm{kSZ}
    renames: aksz
  # Dust priors
  gal545_A_100:
    prior:
      dist: norm
      loc: 8.6
      scale: 2
    ref:
      dist: norm
      loc: 7
      scale: 2
    proposal: 1
    latex: A^\mathrm{dustTT}_{100}
    renames: kgal100
  gal545_A_143:
    prior:
      dist: norm
      loc: 10.6
      scale: 2
    ref:
      dist: norm
      loc: 9
      scale: 2
    proposal: 1
    latex: A^\mathrm{dustTT}_{143}
    renames: kgal143
  gal545_A_143_217:
    prior:
      dist: norm
      loc: 23.5
      scale: 8.5
    ref:
      dist: norm
      loc: 21
      scale: 4
    proposal: 1.5
    latex: A^\mathrm{dustTT}_{\mathrm{143}\times\mathrm{217}}
    renames: kgal143217
  gal545_A_217:
    prior:
      dist: norm
      loc: 91.9
      scale: 20
    ref:
      dist: norm
      loc: 80
      scale: 15
    proposal: 2
    latex: A^\mathrm{dustTT}_{217}
    renames: kgal217
  # Calibration factors
  calib_100T:
    prior:
      dist: norm
      loc: 1.0002
      scale: 0.0007
    ref:
      dist: norm
      loc: 1.0002
      scale: 0.001
    proposal: 0.0005
    latex: c_{100}
    renames: cal0
  calib_217T:
    prior:
      dist: norm
      loc: 0.99805
      scale: 0.00065
    ref:
      dist: norm
      loc: 0.99805
      scale: 0.001
    proposal: 0.0005
    latex: c_{217}
    renames: cal2
  # Subpixel effect
  A_sbpx_100_100_TT: 1
  A_sbpx_143_143_TT: 1
  A_sbpx_143_217_TT: 1
  A_sbpx_217_217_TT: 1
  # Point source contributions
  ps_A_100_100:
    prior:
      dist: uniform
      min: 0
      max: 400
    ref:
      dist: norm
      loc: 257
      scale: 24
    proposal: 17
    latex: A^\mathrm{PS}_{100}
    renames: aps100
  ps_A_143_143:
    prior:
      dist: uniform
      min: 0
      max: 400
    ref:
      dist: norm
      loc: 47
      scale: 10
    proposal: 3
    latex: A^\mathrm{PS}_{143}
    renames: aps143
  ps_A_143_217:
    prior:
      dist: uniform
      min: 0
      max: 400
    ref:
      dist: norm
      loc: 40
      scale: 12
    proposal: 2
    latex: A^\mathrm{PS}_{\mathrm{143}\times\mathrm{217}}
    renames: aps143217
  ps_A_217_217:
    prior:
      dist: uniform
      min: 0
      max: 400
    ref:
      dist: norm
      loc: 104
      scale: 13
    proposal: 2.5
    latex: A^\mathrm{PS}_{217}
    renames: aps217"""

params_2018_TEEE = r"""
  # Dust priors
  galf_TE_index: -2.4
  galf_TE_A_100:
    prior:
      dist: norm
      loc: 0.130
      scale: 0.042
    ref:
      dist: norm
      loc: 0.130
      scale: 0.1
    proposal: 0.1
    latex: A^\mathrm{dustTE}_{100}
    renames: galfTE100
  galf_TE_A_100_143:
    prior:
      dist: norm
      loc: 0.130
      scale: 0.036
    ref:
      dist: norm
      loc: 0.130
      scale: 0.1
    proposal: 0.1
    latex: A^\mathrm{dustTE}_{\mathrm{100}\times\mathrm{143}}
    renames: galfTE100143
  galf_TE_A_100_217:
    prior:
      dist: norm
      loc: 0.46
      scale: 0.09
    ref:
      dist: norm
      loc: 0.46
      scale: 0.10
    proposal: 0.10
    latex: A^\mathrm{dustTE}_{\mathrm{100}\times\mathrm{217}}
    renames: galfTE100217
  galf_TE_A_143:
    prior:
      dist: norm
      loc: 0.207
      scale: 0.072
    ref:
      dist: norm
      loc: 0.207
      scale: 0.100
    proposal: 0.100
    latex: A^\mathrm{dustTE}_{143}
    renames: galfTE143
  galf_TE_A_143_217:
    prior:
      dist: norm
      loc: 0.69
      scale: 0.09
    ref:
      dist: norm
      loc: 0.69
      scale: 0.1
    proposal: 0.1
    latex: A^\mathrm{dustTE}_{\mathrm{143}\times\mathrm{217}}
    renames: galfTE143217
  galf_TE_A_217:
    prior:
      dist: norm
      loc: 1.938
      scale: 0.54
    ref:
      dist: norm
      loc: 1.938
      scale: 0.2
    proposal: 0.2
    latex: A^\mathrm{dustTE}_{217}
    renames: galfTE217
  # EE now recommended to be left to the central values of the stated priors
  galf_EE_index: -2.4
  galf_EE_A_100:
    value: 0.055  # (± 0.014)
    latex: A^\mathrm{dustEE}_{100}
    renames: galfEE100
  galf_EE_A_100_143:
    value: 0.040  # (± 0.010)
    latex: A^\mathrm{dustEE}_{\mathrm{100}\times\mathrm{143}}
    renames: galfEE100143
  galf_EE_A_100_217:
    value: 0.094  # (± 0.023)
    latex: A^\mathrm{dustEE}_{\mathrm{100}\times\mathrm{217}}
    renames: galfEE100217
  galf_EE_A_143:
    value: 0.086  # (± 0.022)
    latex: A^\mathrm{dustEE}_{143}
    renames: galfEE143
  galf_EE_A_143_217:
    value: 0.21  # (± 0.051)
    latex: A^\mathrm{dustEE}_{\mathrm{143}\times\mathrm{217}}
    renames: galfEE143217
  galf_EE_A_217:
    value: 0.70  # (± 0.18)
    latex: A^\mathrm{dustEE}_{217}
    renames: galfEE217
  # Calibration parameters
  A_pol: 1
  calib_100P: 1.021  # (± 0.01)
  calib_143P: 0.966  # (± 0.01)
  calib_217P: 1.040  # (± 0.01)
  # EE End2End correlated noise
  A_cnoise_e2e_100_100_EE: 1
  A_cnoise_e2e_143_143_EE: 1
  A_cnoise_e2e_217_217_EE: 1
  # Subpixel effect
  A_sbpx_100_100_EE: 1
  A_sbpx_100_143_EE: 1
  A_sbpx_100_217_EE: 1
  A_sbpx_143_143_EE: 1
  A_sbpx_143_217_EE: 1
  A_sbpx_217_217_EE: 1"""

nuisance_priors_2015_TT = """
  SZ: "lambda ksz_norm, A_sz: stats.norm.logpdf(ksz_norm+1.6*A_sz, loc=9.5, scale=3.0)"
"""

nuisance_priors_2018_TT = nuisance_priors_2015_TT

nuisance_2015_params = {
    "calib": params_2015_calib, "TT": params_2015_TT, "TEEE": params_2015_TEEE}

nuisance_2018_params = {
    "calib": params_2018_calib, "TT": params_2018_TT, "TEEE": params_2018_TEEE}

nuisance_2015_priors = {"TT": nuisance_priors_2015_TT}

nuisance_2018_priors = {"TT": nuisance_priors_2018_TT}

nuisance_params = {"2015": nuisance_2015_params, "2018": nuisance_2018_params}
nuisance_priors = {"2015": nuisance_2015_priors, "2018": nuisance_2018_priors}
