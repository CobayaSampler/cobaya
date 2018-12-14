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
import six

# Global
import os
import numpy as np
import pandas as pd
from getdist import MCSamples
import logging

# Local
from cobaya.conventions import _weight, _chi2, _minuslogpost, _minuslogprior
from cobaya.conventions import _separator
from cobaya.log import HandledException

# Suppress getdist output
from getdist import chains

chains.print_load_details = False

# Default chunk size for enlarging collections more efficiently
# If a factor is defined (as a fraction !=0), it is used instead
# (e.g. 0.10 to grow the number of rows by 10%)
enlargement_size = 100
enlargement_factor = None


# Make sure that we don't reach the empty part of the dataframe
def check_end(end, imax):
    if end > imax:
        raise ValueError("Trying to access a sample index larger than "
                         "the amount of samples (%d)!" % imax)


class Collection(object):

    def __init__(self, model, output=None,
                 initial_size=enlargement_size, name=None, extension=None,
                 resuming=False):
        self.name = name
        self.log = logging.getLogger(
            "collection:" + name if name else self.__class__.__name__)
        self.sampled_params = list(model.parameterization.sampled_params())
        self.derived_params = list(model.parameterization.derived_params())
        # Create the dataframe structure
        columns = [_weight, _minuslogpost]
        columns += list(self.sampled_params)
        columns += list(self.derived_params)
        self.prior_names = [
            _minuslogprior + _separator + piname for piname in list(model.prior)]
        columns += [_minuslogprior] + self.prior_names
        self.chi2_names = [_chi2 + _separator + likname for likname in model.likelihood]
        columns += [_chi2] + self.chi2_names
        # Create/load the main data frame and the tracking indices
        if output:
            self.file_name, self.driver = output.prepare_collection(
                name=self.name, extension=extension)
        else:
            self.driver = "dummy"
        if resuming:
            if output:
                try:
                    self._out_load()
                    if set(self.data.columns) != set(columns):
                        self.log.error(
                            "Unexpected column names!\nLoaded: %s\nShould be: %s",
                            list(self.data.columns), columns)
                        raise HandledException
                    self._n = self.data.shape[0]
                    self._n_last_out = self._n
                except IOError:
                    self.log.info(
                        "Could not find a chain to resume. Maybe burn-in didn't finish. "
                        "Creating new chain file!")
                    resuming = False
            else:
                self.log.error("No continuation possible if there is no output.")
                raise HandledException
        else:
            self._out_delete()
        if not resuming:
            self.reset(columns=columns, index=range(initial_size))
            # TODO: the following 2 lines should go into the `reset` method.
            if output:
                self._n_last_out = 0

    def reset(self, columns=None, index=None):
        """Create/reset the DataFrame."""
        if columns is None:
            columns = self.data.columns
        if index is None:
            index = self.data.index
        self.data = pd.DataFrame(np.nan, columns=columns, index=index)
        self._n = 0

    def add(self,
            values, derived=None, weight=1, logpost=None, logpriors=None, loglikes=None):
        self._enlarge_if_needed()
        self.data.at[self._n, _weight] = weight
        if logpost is None:
            try:
                logpost = sum(logpriors) + sum(loglikes)
            except ValueError:
                self.log.error("If a log-posterior is not specified, you need to pass "
                               "a log-likelihood and a log-prior.")
                raise HandledException
        self.data.at[self._n, _minuslogpost] = -logpost
        if logpriors is not None:
            for name, value in zip(self.prior_names, logpriors):
                self.data.at[self._n, name] = -value
            self.data.at[self._n, _minuslogprior] = -sum(logpriors)
        if loglikes is not None:
            for name, value in zip(self.chi2_names, loglikes):
                self.data.at[self._n, name] = -2 * value
            self.data.at[self._n, _chi2] = -2 * sum(loglikes)
        for name, value in zip(self.sampled_params, values):
            self.data.at[self._n, name] = value
        if derived is not None:
            for name, value in zip(self.derived_params, derived):
                self.data.at[self._n, name] = value
        self._n += 1

    def _enlarge_if_needed(self):
        if self._n >= self.data.shape[0]:
            if enlargement_factor:
                enlarge_by = self.data.shape[0] * enlargement_factor
            else:
                enlarge_by = enlargement_size
            self.data = pd.concat([
                self.data, pd.DataFrame(np.nan, columns=self.data.columns,
                                        index=np.arange(self.n(), self.n() + enlarge_by))])

    # Retrieve-like methods
    def n(self):
        return self._n

    def last(self):
        return self.data.loc[-1]

    def n_last_out(self):
        return self._n_last_out

    # Make the dataframe printable (but only the filled ones!)
    def __repr__(self):
        return self.data[:self.n()].__repr__()

    # Make the dataframe iterable over rows
    def __iter__(self):
        return self.data[:self.n()].iterrows()

    # Accessing the dataframe
    def __getitem__(self, *args):
        """
        This is a hack of the DataFrame __getitem__ in order to never go
        beyond the number of samples.

        NB: returns *views*, not *copies*, so careful when assigning and modifying
        the returned values.

        Errors are ValueError because this is not for internal use in this class,
        but an external interface to other classes and the user.
        """
        if len(args) > 1:
            raise ValueError("Use just one index/column, or use .loc[row, column]. "
                             "(Notice that slices in .loc *include* the last point.)")
        if isinstance(args[0], six.string_types):
            return self.data.iloc[:self._n, self.data.columns.get_loc(args[0])]
        elif hasattr(args[0], "__len__"):
            try:
                return self.data.iloc[:self._n,
                       [self.data.columns.get_loc(c) for c in args[0]]]
            except KeyError:
                raise ValueError("Some of the indices are not valid columns.")
        elif isinstance(args[0], six.integer_types):
            check_end(args[0], self._n)
            return self.data.iloc[args[0]]
        elif isinstance(args[0], slice):
            check_end(args[0].stop, self._n)
            return self.data.iloc[args[0]]
        else:
            raise ValueError("Index type not recognized: use column names or slices.")

    # Statistical computations
    def mean(self, first=None, last=None, derived=False):
        """
        Returns the (weighted) mean of the parameters in the chain,
        between `first` (default 0) and `last` (default last obtained),
        optionally including derived parameters if `derived=True` (default `False`).
        """
        return np.average(
            self[list(self.sampled_params) +
                 (list(self.derived_params) if derived else [])]
            [first:last].T,
            weights=self[_weight][first:last], axis=-1)

    def cov(self, first=None, last=None, derived=False):
        """
        Returns the (weighted) mean of the parameters in the chain,
        between `first` (default 0) and `last` (default last obtained),
        optionally including derived parameters if `derived=True` (default `False`).
        """
        weights = (lambda w: (
            {"fweights": w} if np.allclose(np.round(w), w) else {"aweights": w}))(
                self[_weight][first:last].values)
        return np.atleast_2d(np.cov(
            self[list(self.sampled_params) +
                 (list(self.derived_params) if derived else [])][first:last].T,
            **weights))

    def _sampled_to_getdist_mcsamples(self, first=None, last=None):
        """
        Basic interface with getdist -- internal use only!
        (For analysis and plotting use `getdist.mcsamples.loadCobayaSamples`.)
        """
        names = list(self.sampled_params)
        # No logging of warnings temporarily, so getdist won't complain unnecessarily
        logging.disable(logging.WARNING)
        mcsamples = MCSamples(
            samples=self.data[:self.n()][names].values[first:last],
            weights=self.data[:self.n()][_weight].values[first:last],
            loglikes=self.data[:self.n()][_minuslogpost].values[first:last], names=names)
        logging.disable(logging.NOTSET)
        return mcsamples

    # Saving and updating
    def _get_driver(self, method):
        return getattr(self, method + _separator + self.driver)

    # Load a pre-existing file
    def _out_load(self):
        self._get_driver("_load")()

    # Dump/update/delete collection
    def _out_dump(self):
        self._get_driver("_dump")()

    def _out_update(self):
        self._get_driver("_update")()

    def _out_delete(self):
        self._get_driver("_delete")()

    # txt driver
    def _load__txt(self):
        self.log.info("Loading existing sample from '%s'", self.file_name)
        with open(self.file_name, "r") as inp:
            cols = [a.strip() for a in inp.readline().lstrip("#").split()]
            self.data = pd.read_table(
                inp, sep=" ", header=None, names=cols, comment="#", skipinitialspace=True)

    def _dump__txt(self):
        self._dump_slice__txt(0, self.n())

    def _update__txt(self):
        self._dump_slice__txt(self.n_last_out(), self.n())

    def _dump_slice__txt(self, n_min=None, n_max=None):
        if n_min is None or n_max is None:
            self.log.error("Needs to specify the limit n's to dump.")
            raise HandledException
        if self._n_last_out == n_max:
            return
        self._n_last_out = n_max
        n_float = 8
        with open(self.file_name, "a") as out:
            lines = self.data[n_min:n_max].to_string(
                header=["# " + str(self.data.columns[0])] + list(self.data.columns[1:]),
                index=False, na_rep="nan", justify="right",
                float_format=(lambda x: ("%%.%dg" % n_float) % x))
            # remove header if not first dump
            if n_min:
                lines = "\n".join(lines.split("\n")[1:])
            out.write(lines)
            out.write("\n")

    def _delete__txt(self):
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    # dummy driver
    def _dump__dummy(self):
        pass

    def _update__dummy(self):
        pass

    def _delete__dummy(self):
        pass


class OnePoint(Collection):
    """Wrapper of Collection to hold a single point, e.g. the current point of an MCMC."""

    def __init__(self, *args, **kwargs):
        kwargs["initial_size"] = 1
        Collection.__init__(self, *args, **kwargs)

    def __getitem__(self, columns):
        if isinstance(columns, six.string_types):
            return self.data.values[0, self.data.columns.get_loc(columns)]
        else:
            try:
                return self.data.values[0, [self.data.columns.get_loc(c) for c in columns]]
            except KeyError:
                raise ValueError("Some of the indices are not valid columns.")

    # Resets the counter, so the dataframe never fills up!
    def add(self, *args, **kwargs):
        Collection.add(self, *args, **kwargs)
        self._n = 0

    def increase_weight(self, increase):
        # For some reason, faster than `self.data[_weight] += increase`
        self.data.at[0, _weight] += increase

    def add_to_collection(self, collection):
        """Adds this point at the end of a given collection."""
        collection.add(
            self[self.sampled_params],
            derived=(self[self.derived_params] if self.derived_params else None),
            logpost=-self[_minuslogpost], weight=self[_weight],
            logpriors=-np.array(self[self.prior_names]),
            loglikes=-0.5 * np.array(self[self.chi2_names]))

    # Restore original __repr__ (here, there is only 1 sample)
    def __repr__(self):
        return self.data.__repr__()

    # Dumper changed: since we have made self._n=0, it would print nothing
    def _update__txt(self):
        self._dump_slice__txt(0, 1)
