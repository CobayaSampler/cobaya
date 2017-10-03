"""
.. module:: collection

:Synopsis: Class to store the Montecarlo sample
:Author: Jesus Torrado

Class keeping track of the samples.

Basically, a wrapper around a `pandas.DataFrame`.

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os
import numpy as np
import pandas as pd
from getdist import MCSamples

# Local
from cobaya.conventions import _weight, _chi2, _minuslogpost, _minuslogprior, _derived_pre
from cobaya.conventions import separator
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)


# Default chunk size for enlarging collections more efficiently
# If a factor is defined (as a fraction !=0), it is used instead
# (e.g. 0.10 to grow the number of rows by 10%)
enlargement_size = 100
enlargement_factor = None

class Collection():

    def __init__(self, parametrisation, likelihood, output=None,
                 initial_size=enlargement_size, name=None):
        self.name = name
        self.sampled_params = parametrisation.sampled_params().keys()
        self.derived_params = parametrisation.derived_params().keys()
        # Create the dataframe structure
        columns = [_weight, _minuslogpost]
        columns += list(self.sampled_params)
        columns += [_derived_pre+name for name in self.derived_params]
        columns += [_minuslogprior]
        self.chi2_names = [_chi2+separator+name for name in likelihood]
        columns += [_chi2] + self.chi2_names
        # Create the main data frame and the tracking index
        self.data = pd.DataFrame(np.zeros((initial_size,len(columns))), columns=columns)
        self._n = 0
        # Prepare labels and ranges for interfacing with GetDist
        self.labels = parametrisation.labels()


        # RANGES HERE!!!!
        

        # OUTPUT: Driver, file_name, index to track last sample written
        if output:
            self._n_last_out = 0
            self.file_name, self.driver = output.prepare_collection(name=self.name)
            self.out_prepare()
        else:
            self.driver = "dummy"

    def add(self, values, derived=None,
                  weight=1, logpost=None, logprior=None, logliks=None):
        self.enlarge_if_needed()
        self.data[_weight][self._n] = weight
        if logpost == None:
            try:
                logpost = logprior + sum(logliks)
            except ValueError:
                log.error("If a log-posterior is not specified, you need to pass "
                          "a log-likelihood and a log-prior.")
                raise HandledException
        self.data[_minuslogpost][self._n] = -logpost
        if logprior != None:
            self.data[_minuslogprior][self._n] = -logprior
        if logliks is not None:
            for name, value in zip(self.chi2_names, logliks):
                self.data[name][self._n] = -2*value
            self.data[_chi2][self._n] = sum(self.data[self.chi2_names].values[self._n])
        for name, value in zip(self.sampled_params, values):
            self.data[name][self._n] = value
        if derived is not None:
            for name, value in zip(self.derived_params, derived):
                self.data[_derived_pre+name][self._n] = value
        self._n += 1

    def enlarge_if_needed(self):
        if self._n >= self.data.shape[0]:
            if enlargement_factor:
                enlarge_by = self.data.shape[0]*enlargement_factor
            else:
                enlarge_by = enlargement_size
            self.data = pd.concat([self.data,
                pd.DataFrame(np.zeros((enlarge_by, self.data.shape[1])), 
                             columns=self.data.columns,
                             index=np.arange(self.n(),self.n()+enlarge_by))])
        
    # Retrieve-like methods
    def n(self):
        return self._n

    def last(self):
        return self.data.loc[-1]

    def n_last_out(self):
        return self._n_last_out

    # Make the dataframe accesible directly
    def __getitem__(self, *args):
        return self.data[:self.n()].__getitem__(*args)

    # Make the dataframe printable (but only the filled ones!)
    def __repr__(self):
        return self.data[:self.n()].__repr__()

    # Make the dataframe iterable over rows
    def __iter__(self):
        return self.data[:self.n()].iterrows()

    # Statistical computations
    def mean(self, first=None, last=None, derived=False):
        """
        Returns the (weighted) mean of the parameters in the chain,
        between `first` (default 0) and `last` (default last obtained),
        optionally including derived parameters if `derived=True` (default `False`).
        """
        return np.average(
            self[list(self.sampled_params)
                 +(list(self.derived_params) if derived else [])]
                [first:last].T,
            weights=self[_weight][first:last], axis=-1)

    def cov(self, first=None, last=None, derived=False):
        """
        Returns the (weighted) mean of the parameters in the chain,
        between `first` (default 0) and `last` (default last obtained),
        optionally including derived parameters if `derived=True` (default `False`).
        """
        return np.atleast_2d(np.cov(
            self[list(self.sampled_params)
                 +(list(self.derived_params) if derived else [])]
                [first:last].T,
            fweights=self[_weight][first:last]))

    # Saving and updating
    def get_driver(self, method):
        return getattr(self, method+separator+self.driver)

    # Create the file, write the header, etc.
    def out_prepare(self):
        self.get_driver("prepare")()
    # Dump/update collection
    def out_dump(self):
        self.get_driver("dump")()
    def out_update(self):
        self.get_driver("update")()

    # txt driver
    def prepare__txt(self):
        log.info("Sample collection to be written on '%s'", self.file_name)
        header = "#" + self.data[:0].to_csv(sep=" ", header=True)
        with open(self.file_name, "w") as out:
            out.write(header)
    def dump__txt(self):
        self.dump_slice__txt(0, self.n())
    def update__txt(self):
        self.dump_slice__txt(self.n_last_out(), self.n())
    def dump_slice__txt(self, n_min=None, n_max=None):
        if n_min == None or n_max == None:
            log.error("Needs to specify the limit n's to dump.")
            raise HandledException
        self._n_last_out = n_max
        self.data[n_min:n_max].to_csv(self.file_name, sep=" ", na_rep="nan", header=False,
                                      float_format="%.8g", index=False, mode="a")

    # dummy driver
    def prepare__dummy(self):
        pass
    def dump__dummy(self):
        pass
    def update__dummy(self):
        pass

    # Interface with getdist
    def as_getdist_mcsamples(self, derived=True, prior_and_lik=True,
                             first=None, last=None):
        # get names and labels (n.b.: getdist forcefully adds its own $'s)
        # sampled
        names = self.sampled_params + [p+"*" for p in self.derived_params]
        labels = self.labels.values()
        ranges = []
#        ranges = dict([(p,ls) for p,ls in zip(self.prior.names(),self.prior.limits())])
#         ranges.update(dict([(p,(self.likelihood.updated_info_params()[p].get("min"),
#                                    self.likelihood.updated_info_params()[p].get("max")))
#                                for p in self.likelihood.derived]))
        # prior and likelihood
        if prior_and_lik:
            names += [_minuslogprior+"*"] + [_chi2+"*"] + list(self.chi2_names)
            labels += ([r"-\log\pi", r"\chi^2"] +
                       [r"\chi^2_\mathrm{"+name+r"}" for name in self.chi2_names])
        # gather samples and create the getdist sample
        columns = [(("" if n.rstrip("*") not in self.derived_params else _derived_pre)+
                    n.rstrip("*")) for n in names]
        samples = self.data[:self.n()].as_matrix(columns=columns)
        return MCSamples(samples=samples[first:last],
                         weights=self.data[:self.n()][_weight].values[first:last],
                         loglikes=self.data[:self.n()][_minuslogpost].values[first:last],
                         names=names, labels=labels, ranges=ranges)
        
class OnePoint(Collection):
    """Wrapper of Collection to hold a single point, e.g. the current point of an MCMC."""
    def __init__(self, *args, **kwargs):
        kwargs["initial_size"] = 1
        Collection.__init__(self, *args, **kwargs)

    def __getitem__(self, columns):
        if isinstance(columns, basestring):
            columns = [columns]
        return self.data.as_matrix(columns=columns)[0]

    # Resets the counter, so the dataframe never fills up!
    def add(self, *args, **kwargs):
        Collection.add(self, *args, **kwargs)
        self._n = 0

    def increase_weight(self, increase):
        self.data["weight"] += increase
        
    def add_to_collection(self, collection):
        """Adds this point at the end of a given collection."""
        collection.add(
            self[self.sampled_params],
            derived=(self[[_derived_pre+p for p in self.derived_params]]
                     if self.derived_params else None),
            logpost=-self[_minuslogpost][0], weight=self[_weight][0],
            logprior=-self[_minuslogprior][0],
            logliks=-0.5*np.array(self[self.chi2_names]))
    
    # Make the dataframe printable (but only the filled ones!)
    def __repr__(self):
        return self.data[:1].__repr__()
