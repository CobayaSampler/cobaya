"""
.. module:: collection

:Synopsis: Classes to store the Montecarlo samples and single points.
:Author: Jesus Torrado and Antony Lewis

"""

# Global
import os
import logging
import numpy as np
import pandas as pd
from typing import Optional
from getdist import MCSamples
from copy import deepcopy

# Local
from cobaya.conventions import _weight, _chi2, _minuslogpost, _minuslogprior, \
    _get_chi2_name, _separator
from cobaya.tools import load_DataFrame
from cobaya.log import LoggedError, HasLogger

# Suppress getdist output
from getdist import chains

chains.print_load_details = False

# Default chunk size for enlarging collections more efficiently
# If a factor is defined (as a fraction !=0), it is used instead
# (e.g. 0.10 to grow the number of rows by 10%)
enlargement_size: int = 100
enlargement_factor: Optional[int] = None


# Make sure that we don't reach the empty part of the dataframe
def check_index(i, imax):
    if (i > 0 and i >= imax) or (i < 0 and -i > imax):
        raise IndexError("Trying to access a sample index larger than "
                         "the amount of samples (%d)!" % imax)
    if i < 0:
        return imax + i
    return i


# Notice that slices are never supposed to raise IndexError, but an empty list at worst!
def check_slice(ij, imax):
    newlims = {"start": ij.start, "stop": ij.stop}
    if ij.start is None:
        newlims["start"] = 0
    if ij.stop is None:
        newlims["stop"] = imax
    for limname, lim in newlims.items():
        if lim >= 0:
            newlims[limname] = min(imax, lim)
        else:
            newlims[limname] = imax + lim
    return slice(newlims["start"], newlims["stop"], ij.step)


class BaseCollection(HasLogger):
    def __init__(self, model, name=None):
        self.name = name
        self.set_logger(name)
        self.sampled_params = list(model.parameterization.sampled_params())
        self.derived_params = list(model.parameterization.derived_params())
        self.minuslogprior_names = [
            _minuslogprior + _separator + piname for piname in list(model.prior)]
        self.chi2_names = [_get_chi2_name(likname) for likname in model.likelihood]
        columns = [_weight, _minuslogpost]
        columns += list(self.sampled_params)
        # Just in case: ignore derived names as likelihoods: would be duplicate cols
        columns += [p for p in self.derived_params if p not in self.chi2_names]
        columns += [_minuslogprior] + self.minuslogprior_names
        columns += [_chi2] + self.chi2_names
        self.columns = columns


class Collection(BaseCollection):
    """
    Holds a collection of samples, stored internally into a ``pandas.DataFrame``.

    The DataFrame itself is accessible as the ``Collection.data`` attribute, but slicing
    can be done on the ``Collection`` itself.
    """

    def __init__(self, model, output=None,
                 initial_size=enlargement_size, name=None, extension=None, file_name=None,
                 resuming=False, load=False, onload_skip=0, onload_thin=1):
        super().__init__(model, name)
        self._value_dict = {p: np.nan for p in self.columns}
        # Create/load the main data frame and the tracking indices
        # Create the DataFrame structure
        if output:
            self.file_name, self.driver = output.prepare_collection(
                name=self.name, extension=extension)
            if file_name:
                self.file_name = file_name
        else:
            self.driver = "dummy"
        if resuming or load:
            if output:
                try:
                    self._out_load(skip=onload_skip, thin=onload_thin)
                    if set(self.data.columns) != set(self.columns):
                        raise LoggedError(
                            self.log,
                            "Unexpected column names!\nLoaded: %s\nShould be: %s",
                            list(self.data.columns), self.columns)
                    self._n = self.data.shape[0]
                    self._n_last_out = self._n
                except IOError:
                    if resuming:
                        self.log.info(
                            "Could not find a chain to resume. "
                            "Maybe burn-in didn't finish. Creating new chain file!")
                        resuming = False
                    elif load:
                        raise
            else:
                raise LoggedError(self.log,
                                  "No continuation possible if there is no output.")
        else:
            self._out_delete()
        if not resuming and not load:
            self.reset(columns=self.columns, index=range(initial_size))
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
        self._add_dict(self._value_dict, values, derived, weight, logpost, logpriors,
                       loglikes)
        self.data.iloc[self._n] = np.array(list(self._value_dict.values()))
        self._n += 1

    def _add_dict(self, dic, values, derived=None, weight=1, logpost=None, logpriors=None,
                  loglikes=None):
        dic[_weight] = weight
        if logpost is None:
            try:
                logpost = sum(logpriors) + sum(loglikes)
            except ValueError:
                raise LoggedError(
                    self.log, "If a log-posterior is not specified, you need to pass "
                              "a log-likelihood and a log-prior.")
        dic[_minuslogpost] = -logpost
        if logpriors is not None:
            for name, value in zip(self.minuslogprior_names, logpriors):
                dic[name] = -value
            dic[_minuslogprior] = -sum(logpriors)
        if loglikes is not None:
            for name, value in zip(self.chi2_names, loglikes):
                dic[name] = -2 * value
            dic[_chi2] = -2 * sum(loglikes)
        if len(values) != len(self.sampled_params):
            raise LoggedError(
                self.log, "Got %d values for the sampled parameters. Should be %d.",
                len(values), len(self.sampled_params))
        for name, value in zip(self.sampled_params, values):
            dic[name] = value
        if derived is not None:
            if len(derived) != len(self.derived_params):
                raise LoggedError(
                    self.log, "Got %d values for the derived parameters. Should be %d.",
                    len(derived), len(self.derived_params))
            for name, value in zip(self.derived_params, derived):
                dic[name] = value

    def _enlarge_if_needed(self):
        if self._n >= self.data.shape[0]:
            if enlargement_factor:
                enlarge_by = self.data.shape[0] * enlargement_factor
            else:
                enlarge_by = enlargement_size
            self.data = pd.concat([
                self.data, pd.DataFrame(np.nan, columns=self.data.columns,
                                        index=np.arange(len(self),
                                                        len(self) + enlarge_by))])

    def append(self, collection):
        """
        Append another collection.
        Internal method: does not check for consistency!
        """
        self.data = pd.concat([self.data[:len(self)], collection.data], ignore_index=True)
        self._n = len(self) + len(collection)

    # Retrieve-like methods
    # MARKED FOR DEPRECATION IN v3.0
    def n(self):
        self.log.warning("*DEPRECATION*: `Collection.n()` will be deprecated soon "
                         "in favor of `len(Collection)`")
        # BEHAVIOUR TO BE REPLACED BY ERROR:
        return len(self)
    # END OF DEPRECATION BLOCK

    def __len__(self):
        return self._n

    def n_last_out(self):
        return self._n_last_out

    # Make the dataframe printable (but only the filled ones!)
    def __repr__(self):
        return self.data[:len(self)].__repr__()

    # Make the dataframe iterable over rows
    def __iter__(self):
        return self.data[:len(self)].iterrows()

    # Accessing the dataframe
    def __getitem__(self, *args):
        """
        This is a hack of the DataFrame __getitem__ in order to never go
        beyond the number of samples.

        When slicing (e.g. [ini:end:step]) or single index, returns a copy of the
        collection.

        When asking for specific columns, returns a *view* of the original DataFrame
        (so careful when assigning and modifying returned values).

        Errors are ValueError because this is not for internal use in this class,
        but an external interface to other classes and the user.
        """
        if len(args) > 1:
            raise ValueError("Use just one index/column, or use .loc[row, column]. "
                             "(Notice that slices in .loc *include* the last point.)")
        if isinstance(args[0], str):
            return self.data.iloc[:self._n, self.data.columns.get_loc(args[0])]
        elif hasattr(args[0], "__len__"):
            try:
                return self.data.iloc[:self._n,
                                          [self.data.columns.get_loc(c) for c in args[0]]]
            except KeyError:
                raise ValueError("Some of the indices are not valid columns.")
        elif isinstance(args[0], int):
            new_data = self.data.iloc[check_index(args[0], self._n)]
        elif isinstance(args[0], slice):
            new_data = self.data.iloc[check_slice(args[0], self._n)]
        else:
            raise ValueError("Index type not recognized: use column names or slices.")
        return self._copy(data=new_data)

    @property
    def values(self):
        self.data.values

    def _copy(self, data=None):
        """
        Returns a copy of the collection.

        If data specified (default: None), the copy returned contains the given data;
        no checks are performed on given data, so use with care (e.g. use with a slice of
        `self.data`).
        """
        current_data = self.data
        if data is None:
            data = current_data
        # Avoids creating a copy of the data, to save memory
        delattr(self, "data")
        self_copy = deepcopy(self)
        setattr(self, "data", current_data)
        setattr(self_copy, "data", data)
        setattr(self_copy, "_n", len(data))
        return self_copy

    # Dummy function to avoid exposing `data` kwarg, since no checks are performed on it.
    def copy(self, data=None):
        """
        Returns a copy of the collection.
        """
        return self._copy()

    # Statistical computations
    def mean(self, first=None, last=None, derived=False, pweight=False):
        """
        Returns the (weighted) mean of the parameters in the chain,
        between `first` (default 0) and `last` (default last obtained),
        optionally including derived parameters if `derived=True` (default `False`).

        If `pweight=True` (default `False`) weights every point with its probability.
        The estimate of the mean in this case is unstable; use carefully.
        """
        if pweight:
            logps = -self[_minuslogpost][first:last].values.copy()
            logps -= max(logps)
            weights = np.exp(logps)
        else:
            weights = self[_weight][first:last].values
        return np.average(
            self[list(self.sampled_params) +
                 (list(self.derived_params) if derived else [])]
            [first:last].T,
            weights=weights, axis=-1)

    def cov(self, first=None, last=None, derived=False, pweight=False):
        """
        Returns the (weighted) covariance matrix of the parameters in the chain,
        between `first` (default 0) and `last` (default last obtained),
        optionally including derived parameters if `derived=True` (default `False`).

        If `pweight=True` (default `False`) weights every point with its probability.
        The estimate of the covariance matrix in this case is unstable; use carefully.
        """
        if pweight:
            logps = -self[_minuslogpost][first:last].values.copy()
            logps -= max(logps)
            weights = np.exp(logps)
            kwarg = "aweights"
        else:
            weights = self[_weight][first:last].values
            kwarg = "fweights" if np.allclose(np.round(weights), weights) else "aweights"
        weights_kwarg = {kwarg: weights}
        return np.atleast_2d(np.cov(
            self[list(self.sampled_params) +
                 (list(self.derived_params) if derived else [])][first:last].T,
            **weights_kwarg))
        return np.atleast_2d(np.cov(
            self[list(self.sampled_params) +
                 (list(self.derived_params) if derived else [])][first:last].T,
            **weights))

    def bestfit(self):
        """Best fit (maximum likelihood) sample. Returns a copy."""
        return self.data.loc[self.data[_chi2].idxmin()].copy()

    def MAP(self):
        """Maximum-a-posteriori (MAP) sample. Returns a copy."""
        return self.data.loc[self.data[_minuslogpost].idxmin()].copy()

    def _sampled_to_getdist_mcsamples(self, first=None, last=None):
        """
        Basic interface with getdist -- internal use only!
        (For analysis and plotting use `getdist.mcsamples.MCSamplesFromCobaya
        <https://getdist.readthedocs.io/en/latest/mcsamples.html#getdist.mcsamples.loadMCSamples>`_.)
        """
        names = list(self.sampled_params)
        # No logging of warnings temporarily, so getdist won't complain unnecessarily
        logging.disable(logging.WARNING)
        mcsamples = MCSamples(
            samples=self.data[:len(self)][names].values[first:last],
            weights=self.data[:len(self)][_weight].values[first:last],
            loglikes=self.data[:len(self)][_minuslogpost].values[first:last], names=names)
        logging.disable(logging.NOTSET)
        return mcsamples

    # Saving and updating
    def _get_driver(self, method):
        return getattr(self, method + _separator + self.driver)

    # Load a pre-existing file
    def _out_load(self, **kwargs):
        self._get_driver("_load")(**kwargs)

    # Dump/update/delete collection

    def out_update(self):
        self._get_driver("_update")()

    def _out_delete(self):
        self._get_driver("_delete")()

    # txt driver
    def _load__txt(self, skip=0, thin=1):
        self.log.debug("Skipping %d rows and thinning with factor %d.", skip, thin)
        self.data = load_DataFrame(self.file_name, skip=skip, thin=thin)
        self.log.info("Loaded %d samples from '%s'", len(self.data), self.file_name)

    def _dump__txt(self):
        self._dump_slice__txt(0, len(self))

    def _update__txt(self):
        self._dump_slice__txt(self.n_last_out(), len(self))

    def _dump_slice__txt(self, n_min=None, n_max=None):
        if n_min is None or n_max is None:
            raise LoggedError(self.log, "Needs to specify the limit n's to dump.")
        if self._n_last_out == n_max:
            return
        self._n_last_out = n_max
        if not getattr(self, "_txt_formatters", False):
            n_float = 8
            # Add to this 7 places: sign, leading 0's, exp with sign and 3 figures.

            def width_col(col):
                return max(7 + n_float, len(col))

            fmts = ["{:" + "{}.{}".format(width_col(col), n_float) + "g}"
                    for col in self.data.columns]
            # `fmt` as a kwarg with default value is needed to force substitution of var.
            # lambda is defined as a string to allow picklability (also header formatter)
            self._txt_formatters = {
                col: eval("lambda x, fmt=fmt: fmt.format(x)")
                for col, fmt in zip(self.data.columns, fmts)}
            self._header_formatter = [
                eval(
                    'lambda s, w=width_col(col): ("{:>" + "{}".format(w) + "s}").format(s)',
                    {'width_col': width_col, 'col': col})
                for col in self.data.columns]
        do_header = not n_min
        if do_header:
            with open(self.file_name, "a", encoding="utf-8") as out:
                out.write("#" + " ".join(
                    f(col) for f, col
                    in zip(self._header_formatter, self.data.columns))[1:] + "\n")
        with open(self.file_name, "a", encoding="utf-8") as out:
            lines = self.data[n_min:n_max].to_string(
                header=False, index=False, na_rep="nan", justify="right",
                formatters=self._txt_formatters)
            out.write(lines + "\n")

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

    # Make it picklable -- formatters are deleted
    # (they will be generated next time txt is dumped)
    def __getstate__(self):
        attributes = super().__getstate__().copy()
        for attr in ['_txt_formatters', '_header_formatter']:
            try:
                del attributes[attr]
            except KeyError:
                pass
        return attributes


class OneSamplePoint:
    """Wrapper to hold a single point, e.g. the current point of an MCMC.
    Alternative to :class:`~collection.OnePoint`, faster but with less functionality."""

    def __init__(self, model, output_thin=1):
        self.sampled_params = list(model.parameterization.sampled_params())
        self.output_thin = output_thin
        self._added_weight = 0

    def add(self, values, weight=1, **kwargs):
        self.values = values
        self.kwargs = kwargs
        self.weight = weight

    @property
    def logpost(self):
        return self.kwargs['logpost']

    def add_to_collection(self, collection):
        """Adds this point at the end of a given collection."""
        if self.output_thin > 1:
            self._added_weight += self.weight
            if self._added_weight >= self.output_thin:
                weight = self._added_weight // self.output_thin
                self._added_weight %= self.output_thin
            else:
                return False
        else:
            weight = self.weight
        collection.add(self.values, weight=weight, **self.kwargs)
        return True

    def __str__(self):
        return ", ".join(
            ['%s:%.7g' % (k, v) for k, v in zip(self.sampled_params, self.values)])


class OnePoint(Collection):
    """Wrapper of :class:`~collection.Collection` to hold a single point,
    e.g. the current point of an MCMC."""

    def __init__(self, *args, **kwargs):
        kwargs["initial_size"] = 1
        super().__init__(*args, **kwargs)

    def __getitem__(self, columns):
        if isinstance(columns, str):
            return self.data.values[0, self.data.columns.get_loc(columns)]
        else:
            try:
                return self.data.values[
                    0, [self.data.columns.get_loc(c) for c in columns]]
            except KeyError:
                raise ValueError("Some of the indices are not valid columns.")

    # Resets the counter, so the DataFrame never fills up!
    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        self._n = 0

    def increase_weight(self, increase):
        # For some reason, faster than `self.data[_weight] += increase`
        self.data.at[0, _weight] += increase

    # Restore original __repr__ (here, there is only 1 sample)
    def __repr__(self):
        return self.data.__repr__()

    # Dumper changed: since we have made self._n=0, it would print nothing
    def _update__txt(self):
        self._dump_slice__txt(0, 1)
