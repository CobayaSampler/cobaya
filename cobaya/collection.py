"""
.. module:: collection

:Synopsis: Classes to store the Montecarlo samples and single points.
:Author: Jesus Torrado and Antony Lewis

"""

# Global
import os
import functools
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Union, Sequence, Optional
from getdist import MCSamples, chains  # type: ignore

# Local
from cobaya.conventions import OutPar, minuslogprior_names, chi2_names, \
    derived_par_name_separator
from cobaya.tools import load_DataFrame
from cobaya.log import LoggedError, HasLogger, NoLogging
from cobaya.model import LogPosterior

# Suppress getdist output
chains.print_load_details = False

# Size of fast numpy cache
# (used to avoid "setting" in Pandas too often, which is expensive)
_default_cache_size = 200


# Make sure that we don't reach the empty part of the dataframe
def check_index(i, imax):
    if (i > 0 and i >= imax) or (i < 0 and -i > imax):
        raise IndexError("Trying to access a sample index larger than "
                         "the amount of samples (%d)!" % imax)
    if i < 0:
        return imax + i
    return i


# Notice that slices are never supposed to raise IndexError, but an empty list at worst!
def check_slice(ij: slice, imax=None):
    newlims = {"start": ij.start, "stop": ij.stop}
    if ij.start is None:
        newlims["start"] = 0
    if ij.stop is None and imax is not None:
        newlims["stop"] = imax
    if imax is not None:
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
        self.minuslogprior_names = minuslogprior_names(model.prior)
        self.chi2_names = chi2_names(model.likelihood)
        columns = [OutPar.weight, OutPar.minuslogpost]
        columns += list(self.sampled_params)
        # Just in case: ignore derived names as likelihoods: would be duplicate cols
        columns += [p for p in self.derived_params if p not in self.chi2_names]
        columns += [OutPar.minuslogprior] + self.minuslogprior_names
        columns += [OutPar.chi2] + self.chi2_names
        self.columns = columns


def ensure_cache_dumped(method):
    """
    Decorator for SampleCollection methods that need the cache cleaned before running.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._cache_dump()
        return method(self, *args, **kwargs)

    return wrapper


class SampleCollection(BaseCollection):
    """
    Holds a collection of samples, stored internally into a ``pandas.DataFrame``.

    The DataFrame itself is accessible as the ``SampleCollection.data`` property,
    but slicing can be done on the ``SampleCollection`` itself
    (returns a copy, not a view).

    Note for developers: when expanding this class or inheriting from it, always access
    the underlying DataFrame as ``self.data`` and not ``self._data``, to ensure the cache
    has been dumped. If you really need to access the actual attribute ``self._data`` in a
    method, make sure to decorate it with ``@ensure_cache_dumped``.
    """

    def __init__(self, model, output=None, cache_size=_default_cache_size, name=None,
                 extension=None, file_name=None, resuming=False, load=False,
                 onload_skip=0, onload_thin=1):
        super().__init__(model, name)
        self.cache_size = cache_size
        # Create/load the main data frame and the tracking indices
        # Create the DataFrame structure
        if output:
            if file_name:
                self.file_name = file_name
                self.driver = output.kind
            else:
                self.file_name, self.driver = output.prepare_collection(
                    name=self.name, extension=extension)
            self.root_file_name = os.path.join(output.folder, output.prefix)
        else:
            self.driver = "dummy"
            self.root_file_name = None
        if resuming or load:
            if output:
                try:
                    self._out_load(skip=onload_skip)
                    if onload_thin != 1:
                        self.thin_samples(onload_thin, inplace=True)
                    if load:
                        self.columns = list(self.data.columns)
                        loaded_chi2_names = set(
                            name for name in self.columns
                            if name.startswith(OutPar.chi2 + derived_par_name_separator))
                        loaded_chi2_names.discard(
                            OutPar.chi2 + derived_par_name_separator + 'prior')
                        if set(self.chi2_names).difference(loaded_chi2_names):
                            raise LoggedError(self.log,
                                              "Input samples do not have chi2 values "
                                              "matching likelihoods in the model:\n "
                                              "found: %s\nexpected: %s\n",
                                              loaded_chi2_names, self.chi2_names)
                        unexpected = loaded_chi2_names.difference(
                            self.chi2_names).difference(self.derived_params)
                        if unexpected:
                            raise LoggedError(self.log,
                                              "Input samples have chi2 values "
                                              "that are not expected: %s ", unexpected)
                    else:
                        data_col_set = set(self.data.columns)
                        col_set = set(self.columns)
                        if data_col_set != col_set:
                            raise LoggedError(
                                self.log,
                                "Unexpected column names!\nLoaded: %s\nShould be: %s",
                                list(self.data.columns), self.columns)
                    self._n_last_out = len(self)
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
            self.reset()
        # Prepare fast numpy cache
        self._icol = {col: i for i, col in enumerate(self.columns)}
        self._cache_reset()

    def reset(self):
        """Create/reset the DataFrame."""
        self._cache_reset()
        self._data = pd.DataFrame(columns=self.columns, dtype=np.float64)
        if getattr(self, "file_name", None):
            self._n_last_out = 0

    def add(self, values: Union[Sequence[float], np.ndarray],
            logpost: Optional[Union[LogPosterior, float]] = None,
            logpriors: Optional[Sequence[float]] = None,
            loglikes: Optional[Sequence[float]] = None,
            derived: Optional[Sequence[float]] = None,
            weight: float = 1):
        """
        Adds a point to the collection.

        If `logpost` can be :class:`~model.LogPosterior`, float or None (in which case,
        `logpriors`, `loglikes` are both required).
        """
        logposterior = self._check_before_adding(
            values, logpost=logpost, logpriors=logpriors,
            loglikes=loglikes, derived=derived, weight=weight)
        self._cache_add(values, logposterior=logposterior, weight=weight)

    def _check_before_adding(self, values: Union[Sequence[float], np.ndarray],
                             logpost: Optional[Union[LogPosterior, float]] = None,
                             logpriors: Optional[Sequence[float]] = None,
                             loglikes: Optional[Sequence[float]] = None,
                             derived: Optional[Sequence[float]] = None,
                             weight: float = 1
                             ) -> LogPosterior:
        """
        Checks that the arguments of collection.add are correctly formatted.

        Returns a :class:`~model.LogPosterior` dataclass (unchanged if one passed).
        """
        if weight is not None and weight <= 0:
            raise LoggedError(self.log, "Weights must be positive. Got %r", weight)
        if len(values) != len(self.sampled_params):
            raise LoggedError(
                self.log, "Got %d values for the sampled parameters. Should be %d.",
                len(values), len(self.sampled_params))
        if derived is not None:
            if len(derived) != len(self.derived_params):
                raise LoggedError(
                    self.log, "Got %d values for the derived parameters. Should be %d.",
                    len(derived), len(self.derived_params))
        if isinstance(logpost, LogPosterior):
            # If priors and likes passed, check consistency
            if logpriors is not None:
                if not np.allclose(logpriors, logpost.logpriors):
                    raise LoggedError(
                        self.log,
                        "logpriors not consistent with LogPosterior object passed.")
            if loglikes is not None:
                if not np.allclose(loglikes, logpost.loglikes):
                    raise LoggedError(
                        self.log,
                        "loglikes not consistent with LogPosterior object passed.")
            if derived is not None:
                # A simple np.allclose is not enough, because np.allclose([1], []) = True!
                if len(derived) != len(logpost.derived) or \
                        not np.allclose(derived, logpost.derived):
                    raise LoggedError(
                        self.log,
                        "derived params not consistent with those of LogPosterior object "
                        "passed.")
            return_logpost = logpost
        elif isinstance(logpost, float) or logpost is None:
            try:
                return_logpost = LogPosterior(
                    logpriors=logpriors, loglikes=loglikes, derived=derived)
            except ValueError as valerr:
                # missing logpriors/loglikes if logpost is None,
                # or inconsistent sum if logpost given
                raise LoggedError(self.log, str(valerr))
        else:
            raise LoggedError(
                self.log, "logpost must be a LogPosterior object, a number or None (in "
                          "which case logpriors and loglikes are needed).")
        return return_logpost

    def _cache_reset(self):
        self._cache = np.full((self.cache_size, len(self.columns)), np.nan)
        self._cache_last = -1

    def _cache_add(self, values: Union[Sequence[float], np.ndarray],
                   logposterior: LogPosterior, weight: float = 1):
        """
        Adds the given point to the cache. Dumps and resets the cache if full.
        """
        if self._cache_last == self.cache_size - 1:
            self._cache_dump()
        self._cache_add_row(
            self._cache_last + 1, values, logposterior=logposterior, weight=weight)
        self._cache_last += 1

    def _cache_add_row(self, pos: int, values: Union[Sequence[float], np.ndarray],
                       logposterior: LogPosterior, weight: float = 1):
        """
        Adds the given point to the cache at the given position.
        """
        self._cache[pos, self._icol[OutPar.weight]] = weight if weight is not None else 1
        self._cache[pos, self._icol[OutPar.minuslogpost]] = -logposterior.logpost
        for name, value in zip(self.sampled_params, values):
            self._cache[pos, self._icol[name]] = value
        if logposterior.logpriors is not None:
            for name, value in zip(self.minuslogprior_names, logposterior.logpriors):
                self._cache[pos, self._icol[name]] = -value
            self._cache[pos, self._icol[OutPar.minuslogprior]] = - logposterior.logprior
        if logposterior.loglikes is not None:
            for name, value in zip(self.chi2_names, logposterior.loglikes):
                self._cache[pos, self._icol[name]] = -2 * value
            self._cache[pos, self._icol[OutPar.chi2]] = -2 * logposterior.loglike
        if len(logposterior.derived):
            for name, value in zip(self.derived_params, logposterior.derived):
                self._cache[pos, self._icol[name]] = value

    def _cache_dump(self):
        """
        Dumps the cache into the pandas table (unless empty).
        """
        if self._cache_last == -1:
            return
        self._enlarge(self._cache_last + 1)
        self._data.iloc[len(self._data) - (self._cache_last + 1):len(self._data)] = \
            self._cache[:self._cache_last + 1]
        self._cache_reset()

    def _enlarge(self, n):
        """
        Enlarges the DataFrame by `n` rows.
        """
        self._data = pd.concat([
            self._data, pd.DataFrame(
                np.nan, columns=self._data.columns,
                index=np.arange(len(self._data), len(self._data) + n))])

    def append(self, collection):
        """
        Append another collection.
        Internal method: does not check for consistency!
        """
        self._data = pd.concat([self.data, collection.data],
                               ignore_index=True)

    def __len__(self):
        return len(self._data) + (self._cache_last + 1)

    def n_last_out(self):
        return self._n_last_out

    @property  # type: ignore
    @ensure_cache_dumped
    def data(self):
        return self._data

    # Make the dataframe printable (but only the filled ones!)
    def __repr__(self):
        return self.data.__repr__()

    # Make the dataframe iterable over rows
    def __iter__(self):
        return self.data.iterrows()

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
            return self.data.iloc[:, self.data.columns.get_loc(args[0])]
        elif hasattr(args[0], "__len__"):  # assume list of columns
            try:
                return self.data.iloc[:, [self.data.columns.get_loc(c) for c in args[0]]]
            except KeyError:
                raise ValueError("Some of the indices are not valid columns.")
        elif isinstance(args[0], int):
            new_data = self.data.iloc[check_index(args[0], len(self.data))]
        elif isinstance(args[0], slice):
            new_data = self.data.iloc[check_slice(args[0])]
        else:
            raise ValueError("Index type not recognized: use column names or slices.")
        return self._copy(data=new_data)

    @property
    def values(self) -> np.ndarray:
        return self.data.to_numpy(dtype=np.float64)

    def to_numpy(self, dtype=None, copy=False) -> np.ndarray:
        return self.data.to_numpy(copy=copy, dtype=dtype or np.float64)

    def _copy(self, data=None, empty=False) -> 'SampleCollection':
        """
        Returns a copy of the collection.

        If ``empty=True`` (default ``False``), returns an empty copy.

        If data specified (default: None), the copy returned contains the given data;
        no checks are performed on given data, so use with care (e.g. use with a slice of
        ``self.data``).
        """
        current_data = self.data
        if data is None:
            data = current_data
        # Avoids creating an unnecessary copy of the data, to save memory
        delattr(self, "_data")
        self_copy = deepcopy(self)
        setattr(self, "_data", current_data)
        if empty:
            self_copy.reset()
        else:
            setattr(self_copy, "_data", data.copy())
            setattr(self_copy, "_n", data.last_valid_index() + 1)
        return self_copy

    # Dummy function to avoid exposing `data` kwarg, since no checks are performed on it.
    def copy(self, empty=False) -> 'SampleCollection':
        """
        Returns a copy of the collection.

        If ``empty=True`` (default ``False``), returns an empty copy.
        """
        return self._copy(empty=empty)

    def _weights_for_stats(self, first: Optional[int] = None, last: Optional[int] = None,
                           pweight: bool = False) -> np.ndarray:
        """
        Returns weights for computation of statistical quantities such as mean and
        covariance.

        If ``pweight=True`` (default ``False``) every point is weighted with its
        probability, and sample weights are ignored (may lead to very inaccurate
        estimates!).
        """
        if pweight:
            logps = -self[OutPar.minuslogpost][first:last]\
                .to_numpy(dtype=np.float64, copy=True)
            logps -= max(logps)
            weights = np.exp(logps)
        else:
            weights = self[OutPar.weight][first:last]\
                .to_numpy(dtype=np.float64, copy=True)
        return weights

    def mean(self, first: Optional[int] = None, last: Optional[int] = None,
             pweight: bool = False, derived: bool = False) -> np.ndarray:
        """
        Returns the (weighted) mean of the parameters in the chain,
        between `first` (default 0) and `last` (default last obtained),
        optionally including derived parameters if `derived=True` (default `False`).

        If ``pweight=True`` (default ``False``) every point is weighted with its
        probability, and sample weights are ignored (may lead to inaccurate
        estimates!).
        """
        weights = self._weights_for_stats(first, last, pweight=pweight)
        return np.average(self[list(self.sampled_params) +
                               (list(self.derived_params) if derived else [])]
                          [first:last].to_numpy(dtype=np.float64).T, weights=weights,
                          axis=-1)

    def cov(self, first: Optional[int] = None, last: Optional[int] = None,
            pweight: bool = False, derived: bool = False) -> np.ndarray:
        """
        Returns the (weighted) covariance matrix of the parameters in the chain,
        between `first` (default 0) and `last` (default last obtained),
        optionally including derived parameters if `derived=True` (default `False`).

        If ``pweight=True`` (default ``False``) every point is weighted with its
        probability, and sample weights are ignored (may lead to very inaccurate
        estimates!).
        """
        weights = self._weights_for_stats(first, last, pweight=pweight)
        if pweight:
            weight_type_kwarg = "aweights"
        elif np.allclose(np.round(weights), weights):
            weights = np.round(weights).astype(int)
            weight_type_kwarg = "fweights"
        else:
            weight_type_kwarg = "aweights"
        return np.atleast_2d(np.cov(
            self[list(self.sampled_params) +
                 (list(self.derived_params) if derived else [])][first:last].to_numpy(
                dtype=np.float64).T,
            **{weight_type_kwarg: weights}))

    def reweight(self, importance_weights):
        """
        Reweights the sample with the given ``importance_weights``.

        This cannot be fully undone (e.g. recovering original integer weights).
        You may want to call this method on a copy (see :func:`SampleCollection.copy`).
        """
        self._cache_dump()
        self._data[OutPar.weight] *= importance_weights
        self._data = self.data[self._data.weight > 0].reset_index(drop=True)
        self._n = self._data.last_valid_index() + 1

    def filtered_copy(self, where) -> 'SampleCollection':
        return self._copy(self.data[where].reset_index(drop=True))

    def thin_samples(self, thin, inplace=False) -> 'SampleCollection':
        if thin == 1:
            return self if inplace else self.copy()
        if thin != int(thin) or thin < 1:
            raise LoggedError(self.log, "Thin factor must be a positive integer, got %s",
                              thin)
        from getdist.chains import WeightedSamples, WeightedSampleError  # type: ignore
        thin = int(thin)
        try:
            if hasattr(WeightedSamples, "thin_indices_and_weights"):
                unique, counts = \
                    WeightedSamples.thin_indices_and_weights(
                        thin, self[OutPar.weight].to_numpy(dtype=np.float64))
            else:
                raise LoggedError(self.log, "Thinning requires GetDist 1.2+", )
        except WeightedSampleError as e:
            raise LoggedError(self.log, "Error thinning: %s", e)
        else:
            data = self._data.iloc[unique, :].copy()
            data.iloc[:, 0] = counts
            data.reset_index(drop=True, inplace=True)
            if inplace:
                self._data = data
                self._n = self._data.last_valid_index() + 1
            else:
                return self._copy(data)
        return self

    def bestfit(self):
        """Best fit (maximum likelihood) sample. Returns a copy."""
        return self.data.loc[self.data[OutPar.chi2].idxmin()].copy()

    def MAP(self):
        """Maximum-a-posteriori (MAP) sample. Returns a copy."""
        return self.data.loc[self.data[OutPar.minuslogpost].idxmin()].copy()

    def sampled_to_getdist_mcsamples(
        self, first: Optional[int] = None, last: Optional[int] = None) -> MCSamples:
        """
        Basic interface with getdist -- internal use only!
        (For analysis and plotting use `getdist.mcsamples.MCSamplesFromCobaya
        <https://getdist.readthedocs.io/en/latest/mcsamples.html#getdist.mcsamples.loadMCSamples>`_.)
        """
        names = list(self.sampled_params)
        # No logging of warnings temporarily, so getdist won't complain unnecessarily
        with NoLogging():
            mcsamples = MCSamples(
                samples=self.data[names].to_numpy(dtype=np.float64)[
                        first:last],
                weights=self.data[OutPar.weight].to_numpy(dtype=np.float64)[
                        first:last],
                loglikes=self.data[OutPar.minuslogpost].to_numpy(
                    dtype=np.float64)[first:last],
                names=names)
        return mcsamples

    # Saving and updating
    def _get_driver(self, method):
        return getattr(self, method + derived_par_name_separator + self.driver)

    # Load a pre-existing file
    def _out_load(self, **kwargs):
        self._get_driver("_load")(**kwargs)
        self._cache_reset()

    # Dump/update/delete collection

    def out_update(self):
        self._get_driver("_update")()

    def _out_delete(self):
        self._get_driver("_delete")()

    # txt driver
    def _load__txt(self, skip=0):
        self.log.debug("Skipping %d rows", skip)
        self._data = load_DataFrame(self.file_name, skip=skip,
                                    root_file_name=self.root_file_name)
        self.log.info("Loaded %d sample points from '%s'", len(self._data),
                      self.file_name)

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
        if not hasattr(self, "_numpy_fmts"):
            n_float = 8
            # Add to this 7 places: sign, leading 0's, exp with sign and 3 figures.
            width_col = lambda col: max(7 + n_float, len(col))
            self._numpy_fmts = ["%{}.{}".format(width_col(col), n_float) + "g"
                                for col in self.data.columns]
            self._header_formatter = [
                eval('lambda s, w=width_col(col): '
                     '("{:>" + "{}".format(w) + "s}").format(s)',
                     {'width_col': width_col, 'col': col})
                for col in self.data.columns]
        do_header = not n_min
        if do_header:
            if os.path.exists(self.file_name):
                raise LoggedError(self.log,
                                  "Output file %s already exists (report bug)",
                                  self.file_name)
            with open(self.file_name, "a", encoding="utf-8") as out:
                out.write("#" + " ".join(
                    f(col) for f, col
                    in zip(self._header_formatter, self.data.columns))[1:] + "\n")
        with open(self.file_name, "a", encoding="utf-8") as out:
            np.savetxt(out, self.data[n_min:n_max].to_numpy(dtype=np.float64),
                       fmt=self._numpy_fmts)

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
        for attr in ['_numpy_fmts', '_header_formatter']:
            try:
                del attributes[attr]
            except KeyError:
                pass
        return attributes


class OneSamplePoint:
    """Wrapper to hold a single point, e.g. the current point of an MCMC.
    Alternative to :class:`~collection.OnePoint`, faster but with less functionality."""
    results: LogPosterior
    values: np.ndarray
    weight: int

    def __init__(self, model, output_thin=1):
        self.sampled_params = list(model.parameterization.sampled_params())
        self.output_thin = output_thin
        self._added_weight = 0

    def add(self, values, logpost: LogPosterior):
        self.values = values
        if not isinstance(logpost, LogPosterior):
            raise ValueError("`logpost` argument must be LogPosterior instance.")
        self.results = logpost
        self.weight = 1

    @property
    def logpost(self):
        return self.results.logpost

    def add_to_collection(self, collection: SampleCollection):
        """
        Adds this point at the end of a given collection.

        It is assumed that both this instance and the collection passed were
        initialised with the same :class:`model.Model` (no checks are performed).
        """
        if self.output_thin > 1:
            self._added_weight += self.weight
            if self._added_weight >= self.output_thin:
                weight = self._added_weight // self.output_thin
                self._added_weight %= self.output_thin
            else:
                return False
        else:
            weight = self.weight
        collection.add(self.values, logpost=self.results, weight=weight)
        return True

    def __str__(self):
        return ", ".join(
            ['%s:%.7g' % (k, v) for k, v in zip(self.sampled_params, self.values)])


class OnePoint(SampleCollection):
    """Wrapper of :class:`~collection.SampleCollection` to hold a single point,
    e.g. the best-fit point of a minimization run (not used by default MCMC)."""

    def __getitem__(self, columns):
        if isinstance(columns, str):
            return self.data.values[0, self.data.columns.get_loc(columns)]
        else:
            try:
                return self.data.values[0,
                                        [self.data.columns.get_loc(c) for c in columns]]
            except KeyError:
                raise ValueError("Some of the indices are not valid columns.")

    def add(self, *args, **kwargs):
        self.reset()  # resets the DataFrame, so never goes beyond 1 point
        super().add(*args, **kwargs)

    def increase_weight(self, increase):
        # For some reason, faster than `self.data[Par.weight] += increase`
        self.data.at[0, OutPar.weight] += increase

    # Restore original __repr__ (here, there is only 1 sample)
    def __repr__(self):
        return self.data.__repr__()

    # Dumper changed: always force to print the single element.
    def _update__txt(self):
        self._dump_slice__txt(0, 1)
