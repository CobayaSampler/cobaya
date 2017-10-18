"""
.. module:: planck_clik_prototype

:Synopsis: Definition of the clik code based likelihoods
:Author: Julien Lesgourgues and Benjamin Audren (and Jesus Torrado for small compatibility changes only)

Contains the definition of the base clik 2.0 likelihoods, from which all Planck 2015
likelihoods inherit.

The treatment on Planck 2015 likelihoods has been adapted without much modification from
the `MontePython <http://baudren.github.io/montepython.html>`_ code by Julien Lesgourgues
and Benjamin Audren.

.. note::
  
  An older, `MIT-licensed <https://opensource.org/licenses/MIT>`_ version of this module by
  Julien Lesgourgues and Benjamin Audren can be found in the `source of
  MontePython <https://github.com/baudren/montepython_public>`_.


The Planck 2015 likelihoods defined here are:

- ``planck_2015_lowl``
- ``planck_2015_lowTEB``
- ``planck_2015_plikHM_TT``
- ``planck_2015_plikHM_TTTEEE``
- ``planck_2015_plikHM_TTTEEE_unbinned``
- ``planck_2015_plikHM_TT_unbinned``


Usage
-----

To use the Planck likelihoods, you simply need to mention them in the likelihood blocks,
specifying the path where you have installed them, for example:

.. code-block:: yaml

   likelihood:
     planck_2015_lowTEB:
       path: /path/to/cosmo/likelihoods/planck2015/
     planck_2015_plikHM_TTTEEE:
       path: /path/to/cosmo/likelihoods/planck2015/

This automatically pulls the likelihood parameters into the sampling, with their default
priors. The parameters can be fixed or their priors modified by re-defining them in the
``params: likelihood:`` block.

The default parameter and priors can be found in the ``defaults.yaml`` files in the
folder corresponding to each likelihood. They are not reproduced here because of their
length.

.. [The installation instructions are in the corresponding doc file because they need
..  some unicode characters to show some directory structure]

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os
import sys
import numpy as np
import logging

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import HandledException
from cobaya.conventions import _path_install
from cobaya.tools import get_path_to_installation


# Making sure that the logger has the name of the likelihood, not the prototype
def set_logger(name):
    global log
    log = logging.getLogger(name)


class planck_clik_prototype(Likelihood):

    def initialise(self):
        self.name = self.__class__.__name__
        set_logger(self.name)
        # Importing Planck's clik library (only once!)
        try:
            clik
        except NameError:
            if not self.path:
                path_to_installation = get_path_to_installation()
                if path_to_installation:
                    self.path_clik = os.path.join(
                        path_to_installation, "code", common_path)
                    self.path_data = os.path.join(
                        path_to_installation, "data", common_path)
                else:
                    log.error("No path given to the Planck likelihood. Set the likelihood"
                              " property 'path' or the common property '%s'.",
                              _path_install)
                    raise HandledException
            log.info("[%s] Importing clik from %s", self.name, self.path_clik)
            # test and import clik
            is_installed_clik(self.path_clik, log_and_fail=True, import_it=False)
            import clik
        # Loading the likelihood data
        clik_file = os.path.join(self.path_data, self.clik_file)
        # for lensing, some routines change. Intializing a flag for easier
        # testing of this condition
        if 'lensing' in self.name and 'Planck' in self.name:
            self.lensing = True
        else:
            self.lensing = False
        try:
            if self.lensing:
                self.clik = clik.clik_lensing(clik_file)
                try: 
                    self.l_max = max(self.clik.get_lmax())
                # following 2 lines for compatibility with lensing likelihoods of 2013 and before
                # (then, clik.get_lmax() just returns an integer for lensing likelihoods;
                # this behavior was for clik versions < 10)
                except:
                    self.l_max = self.clik.get_lmax()
            else:
                self.clik = clik.clik(clik_file)
                self.l_max = max(self.clik.get_lmax())
        except clik.lkl.CError:
            log.error(
                "The path to the .clik file for the likelihood "
                "%s was not found where indicated."
                " Note that the default path to search for it is"
                " one directory above the path['clik'] field. You"
                " can change this behaviour in all the "
                "Planck_something.data, to reflect your local configuration, "
                "or alternatively, move your .clik files to this place.", self.name)
            raise HandledException
        except KeyError:
            log.error(
                "In the %s.data file, the field 'clik' of the "
                "path dictionary is expected to be defined. Please make sure"
                " it is the case in you configuration file", self.name)
            raise HandledException
        # Requested spectra
        if self.lensing:
            raise NotImplementedError("Lensing lik not implemented!!!")
            # For lensing, the order would be: **phiphi TT EE BB TE TB EB**
        requested_i = [c=="1" for c in self.clik.get_has_cl()]
        requested_cls =[cl for cl,i in
                        zip(["TT","EE","BB","TE","TB","EB"], requested_i) if i]
        # State requisites to the theory code
        self.theory.needs({"Cl": requested_cls, "l_max": self.l_max})
        self.expected_params = list(self.clik.extra_parameter_names)
        # line added to deal with a bug in planck likelihood release:
        # A_planck called A_Planck in plik_lite
        if "plikHM_lite" in self.name:
            i = self.expected_params.index('A_Planck')
            self.expected_params[i] = 'A_planck'
            log.info("In %s, corrected nuisance parameter name A_Planck to A_planck" % self.name)
        
    def logp(self, **params_values):
        # get Cl's from the theory code
        cl = self.theory.get_cl()
        # testing for lensing
        if self.lensing:
            try:
                length = len(self.clik.get_lmax())
                tot = np.zeros(
                    np.sum(self.clik.get_lmax()) + length +
                    len(self.params()))
            # following 3 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                length = 2
                tot = np.zeros(2*self.l_max+length + len(self.params()))
        else:
            length = len(self.clik.get_has_cl())
            tot = np.zeros(
                np.sum(self.clik.get_lmax()) + length +
                len(self.expected_params))
        # fill with Cl's
        index = 0
        if not self.lensing:
            for i in range(length):
                if (self.clik.get_lmax()[i] > -1):
                    for j in range(self.clik.get_lmax()[i]+1):
                        if (i == 0):
                            tot[index+j] = cl['tt'][j]
                        if (i == 1):
                            tot[index+j] = cl['ee'][j]
                        if (i == 2):
                            tot[index+j] = cl['bb'][j]
                        if (i == 3):
                            tot[index+j] = cl['te'][j]
                        if (i == 4):
                            tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                        if (i == 5):
                            tot[index+j] = 0 #cl['eb'][j] class does not compute eb
                    index += self.clik.get_lmax()[i]+1
        else:
            try:
                for i in range(length):
                    if (self.clik.get_lmax()[i] > -1):
                        for j in range(self.clik.get_lmax()[i]+1):
                            if (i == 0):
                                tot[index+j] = cl['pp'][j]
                            if (i == 1):
                                tot[index+j] = cl['tt'][j]
                            if (i == 2):
                                tot[index+j] = cl['ee'][j]
                            if (i == 3):
                                tot[index+j] = cl['bb'][j]
                            if (i == 4):
                                tot[index+j] = cl['te'][j]
                            if (i == 5):
                                tot[index+j] = 0 #cl['tb'][j] class does not compute tb
                            if (i == 6):
                                tot[index+j] = 0 #cl['eb'][j] class does not compute eb

                        index += self.clik.get_lmax()[i]+1
            # following 8 lines for compatibility with lensing likelihoods of 2013 and before
            # (then, clik.get_lmax() just returns an integer for lensing likelihoods,
            # and the length is always 2 for cl['pp'], cl['tt'])
            except:
                for i in range(length):
                    for j in range(self.l_max):
                        if (i == 0):
                            tot[index+j] = cl['pp'][j]
                        if (i == 1):
                            tot[index+j] = cl['tt'][j]
                    index += self.l_max+1
        # fill with likelihood parameters
        for i,p in enumerate(self.expected_params):
            tot[index+i] = params_values[p]
        # In case there are derived parameters in the future:
        # derived = params_values.get("derived")
        # if derived != None:
        #     derived["whatever"] = [...]
        # Compute the likelihood
        return self.clik(tot)[0]

    def close(self):
        del(self.clik) # MANDATORY: forces deallocation of the Cython class
        # Actually, it does not work for low-l likelihoods, which is quite dangerous!


# Installation routines ##################################################################

# path to be shared by all Planck likelihoods
common_path = "planck_2015"


def download_from_planck(product_id, path):
    try:
        from wget import download, bar_thermometer
        wget_kwargs = {"out": path, "bar": bar_thermometer}
        prefix = r"http://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID="
        filename = download(prefix+product_id, **wget_kwargs)
    except:
        log.error("Error downloading!")
        return False
    finally:
        print ""  # force newline after wget
    # uncompress
    import os
    import tarfile
    extension = os.path.splitext(filename)[-1][1:]
    tar = tarfile.open(filename, "r:"+extension)
    try:
        tar.extractall(path)
        tar.close()
        os.remove(filename)
        return True
    except:
        log.error("Error decompressing downloaded file! Corrupt file?)")
        return False


def is_installed_clik(path, log_and_fail=False, import_it=True):
    clik_path = os.path.join(path, "plc-2.0")
    if not os.path.exists(clik_path):
        if log_and_fail:
            log.error("The given folder does not exist: '%s'",clik_path)
            raise HandledException
        return False
    clik_path = os.path.join(clik_path, "lib/python2.7/site-packages")
    if not os.path.exists(clik_path):
        if log_and_fail:
            log.error("You have not compiled the Planck likelihood code 'clik'.\n"
                      "Take a look at the docs to see how to do it using 'waf'.")
            raise HandledException
        return False
    sys.path.insert(0, clik_path)
    try:
        if import_it:
            import clik
        return True
    except:
        return False


def install_clik(path):
    log.info("clik: downlowading...")
    if not download_from_planck("1904", path):
        log.error("Not possible to download clik.")
        return False
    log.info("clik: configuring... (and maybe installing dependencies...)")
    os.chdir(os.path.join(path, "plc-2.0"))
    from subprocess import Popen, PIPE
    process = Popen(
        ["./waf", "configure", "--install_all_deps"], stdout=PIPE, stderr=PIPE)
    out, err = process.communicate()
    if err or not out.split("\n")[-2].startswith("'configure' finished successfully"):
        print out
        print err
        log.error("Configuration failed!")
        return False
    log.info("clik: compiling...")
    process2 = Popen(["./waf", "install"], stdout=PIPE, stderr=PIPE)
    out2, err2 = process2.communicate()
    # We don't check that err2" is empty, because harmless warnings are included there.
    if not out2.split("\n")[-2].startswith("'install' finished successfully"):
        print out2
        print err2
        log.error("Compilation failed!")
        return False
    log.info("clik: finished!")
    return True


def get_product_id_and_clik_file(name):
    # get it from the defaults.yaml file
    from cobaya.conventions import _defaults_file, _likelihood
    path__defaults_file = os.path.join(
        os.path.dirname(__file__), "..", name, _defaults_file)
    from cobaya.yaml_custom import yaml_load_file
    defaults = yaml_load_file(path__defaults_file)[_likelihood][name]
    return defaults["product_id"], defaults["clik_file"]


def is_installed(**kwargs):
    set_logger(kwargs["name"])
    result = True
    if kwargs["code"]:
        result &= is_installed_clik(os.path.realpath(
            os.path.join(kwargs["path"], "..", "code", common_path)))
    if kwargs["data"]:
        _, filename = get_product_id_and_clik_file(kwargs["name"])
        result &= os.path.exists(os.path.realpath(
            os.path.join(kwargs["path"], "..", "data", common_path, filename)))
    return result


def install(path=None, name=None, force=False, code=True, data=True):
    set_logger(name)
    # Create common folders: all planck likelihoods share install folder for code and data
    paths = {}
    for s in ("code", "data"):
        if eval(s):
            paths[s] = os.path.realpath(os.path.join(path, "..", s, common_path))
            if not os.path.exists(paths[s]):
                os.makedirs(paths[s])
    # Install clik
    if code and (not is_installed_clik(paths["code"]) or force):
        log.info("Installing the clik code.")
        success = install_clik(paths["code"])
        if not success:
            return False
    if data:
        # Extract product_id
        product_id, _ = get_product_id_and_clik_file(name)
        # Download and uncompress the particular likelihood
        log.info("Downloading likelihood data...")
        if not download_from_planck(product_id, paths["data"]):
            log.error("Not possible to download this likelihood.")
            return False
        log.info("Likelihood data downloaded and uncompressed correctly.")
    return True
