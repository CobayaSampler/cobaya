"""
.. module:: collection

:Synopsis: Classes to store the Montecarlo samples and single points.
:Author: Jesus Torrado and Antony Lewis

"""

# Global
import os
import functools
import warnings
from copy import deepcopy
from typing import Union, Sequence, Optional, Tuple, Iterable
from numbers import Number
import numpy as np
import pandas as pd
from getdist import MCSamples, chains  # type: ignore
from getdist.chains import WeightedSamples, WeightedSampleError  # type: ignore

# Local
from cobaya.conventions import OutPar, minuslogprior_names, chi2_names, \
    derived_par_name_separator, minuslogprior_labels, chi2_labels, minuslogpost_label
from cobaya.tools import load_DataFrame
from cobaya.log import LoggedError, HasLogger, NoLogging
from cobaya.model import Model, LogPosterior, DummyModel

# Suppress getdist output
chains.print_load_details = False

# Size of fast numpy cache
# (used to avoid "setting" in Pandas too often, which is expensive)
_default_cache_size = 200

# Sample types, for the purposes of e.g. knowing whether skip/thin operations are allowed.
sample_types = ["mcmc", "nested"]


def check_index(i, imax):
    """Makes sure that we don't reach the empty part of the dataframe."""
    if (i > 0 and i >= imax) or (i < 0 and -i > imax):
        raise IndexError("Trying to access a sample index larger than "
                         f"the amount of samples ({imax})!")
    if i < 0:
        return imax + i
    return i


def check_slice(ij: slice, imax=None):
    """
    Restricts a slice to the non-empty part of the DataFrame.

    Notice that slices are never supposed to raise IndexError, but an empty list at worst!
    """
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


def apply_temperature(logpost, temperature):
    """Applies sampling temperature to a log-posteior."""
    return logpost / temperature


def remove_temperature(logpost, temperature):
    """Removes sampling temperature from a log-posteior."""
    return apply_temperature(logpost, 1 / temperature)


def apply_temperature_cov(cov, temperature):
    """
    Convert covariance matrix to that of ``probability^(1/temperature)`` posterior.
    """
    return cov * temperature


def remove_temperature_cov(cov, temperature):
    """
    Convert covariance matrix of ``probability^(1/temperature)`` posterior to original
    one.
    """
    return cov / temperature


def compute_temperature(logpost, logprior, loglike, check=True):
    """
    Returns the temperature of a sample.

    If ``check=True`` and the log-probabilites passed are arrays, checks consistency
    of the sample temperature, and raises ``AssertionError`` if inconsistent.
    """
    temp = (logprior + loglike) / logpost
    if check and isinstance(temp, Iterable) and len(temp) > 1:
        assert np.allclose(temp, temp[0]), "Inconsistent temperature in sample."
        temp = temp[0]
    return temp


def detempering_weights_factor(tempered_logpost, temperature):
    """
    Returns the detempering factors for the weights of a tempered sample, i.e. if ``w_t``
    is the weight of the tempered sample, then the weight of the unit-temperature one is
    ``w_t * f``, where the ``f`` returned by this method is
    ``exp(logp * (1 - 1/temperature))``, where ``logp`` is the (untempered) logposterior.

    Factors are normalized so that the largest equals one.
    """
    if temperature == 1:
        return np.ones(np.atleast_1d(tempered_logpost).shape)
    log_ratio = remove_temperature(tempered_logpost, temperature) - tempered_logpost
    return np.exp(log_ratio - max(log_ratio))


class BaseCollection(HasLogger):
    def __init__(self, model, name=None, temperature=None):
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
        self.temperature = temperature if temperature is not None else 1
        self.columns = columns
        self._cache_aux_model_quantities(model)

    def _cache_aux_model_quantities(self, model):
        """
        Stores some auxiliary Model-related variables to allow e.g. for interfacing other
        codes without needing to use the Model again.

        Can be called inside these interfaces in case the model has changed.
        """
        self._cached_labels = deepcopy(model.parameterization.labels())
        self._cached_labels[OutPar.minuslogpost] = minuslogpost_label()
        self._cached_labels.update(minuslogprior_labels(model.prior))
        self._cached_labels.update(chi2_labels(model.likelihood))
        self._cached_renames = deepcopy(model.parameterization.sampled_params_renames())
        self._cached_ranges = None
        if not isinstance(model, DummyModel):  # can happen during post-processing
            self._cached_ranges = dict(zip(
                self.sampled_params,
                model.prior.bounds(confidence_for_unbounded=0.9999995)))  # 5 sigmas
        for p, p_info in model.parameterization.derived_params_info().items():
            mini, maxi = p_info.get("min", -np.inf), p_info.get("max", np.inf)
            some_bound_specified = np.isfinite(mini) or np.isfinite(maxi)
            if some_bound_specified:
                self._cached_ranges[p] = [mini, maxi]


def ensure_cache_dumped(method):
    """
    Decorator for SampleCollection methods that need the cache cleaned before running.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._cache_dump()  # pylint: disable=protected-access
        return method(self, *args, **kwargs)

    return wrapper


class SampleCollection(BaseCollection):
    """
    Holds a collection of samples, stored internally into a ``pandas.DataFrame``.

    The DataFrame itself is accessible as the ``SampleCollection.data`` property,
    but slicing can be done on the ``SampleCollection`` itself
    (returns a copy, not a view).

    When ``temperature`` is different from 1, weights and log-posterior (but not priors
    or likelihoods' chi squared's) are those of the tempered sample, obtained assuming a
    posterior raised to the power of ``1/temperature``. Functions returning statistics,
    e.g. :func:`~SampleCollection.cov`, will return the statistics of the original
    (untempered) posterior, unless indicated otherwise with a keyword argument.

    Note for developers: when expanding this class or inheriting from it, always access
    the underlying DataFrame as ``self.data`` and not ``self._data``, to ensure the cache
    has been dumped. If you really need to access the actual attribute ``self._data`` in a
    method, make sure to decorate it with ``@ensure_cache_dumped``.
    """

    def __init__(self, model, output=None, cache_size=_default_cache_size, name=None,
                 extension=None, file_name=None, resuming=False, load=False,
                 temperature=None, onload_skip=0, onload_thin=1, sample_type=None):
        super().__init__(model, name)
        if sample_type is not None and (not isinstance(sample_type, str) or
                                        not sample_type.lower() in sample_types):
            raise LoggedError(self.log, "'sample_type' must be one of %r.", sample_types)
        self.sample_type = sample_type.lower() if sample_type is not None else sample_type
        self.cache_size = cache_size
        self._data = None
        self._n = None
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
                            missing = set(self.columns).difference(self.data.columns)
                            unexpected = set(self.data.columns).difference(self.columns)
                            raise LoggedError(
                                self.log, (f"Bad column names! Missing {missing}; "
                                           f"unexpected: {unexpected}"))
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
        # If loaded, check sample weights, consistent logp sums,
        # and temperature (ignores the given one)
        samples_loaded = len(self) > 0
        if samples_loaded:
            try:
                self.temperature = self._check_logps()
                if temperature is not None and \
                   not np.isclose(temperature, self.temperature):
                    raise LoggedError(
                        self.log,
                        "Sample temperature appears to be %r, but the collection was "
                        "explicitly initialized with temperature %r.",
                        self.temperature,
                        temperature,
                    )
                self._check_weights()
            except LoggedError as excpt:
                raise LoggedError(
                    self.log,
                    "Error when loading samples: %s",
                    str(excpt)
                ) from excpt
            self._drop_samples_null_weight()
            if self.is_tempered and not resuming:
                self.log.warning(
                    "The collection loaded has temperature != 1. "
                    "Keep that in mind when operating on it, or detemper (in-place) with "
                    "'SampleCollection.reset_temperature()'."
                )
        else:
            self.temperature = temperature if temperature is not None else 1
        # Prepare fast numpy cache
        self._icol = {col: i for i, col in enumerate(self.columns)}
        self._cache_reset()
        # Prepare txt formatter
        self.n_float = 8
        # Add to this 7 places: sign, leading 0's, exp with sign and 3 figures.
        width_col = lambda col: max(7 + self.n_float, len(col))
        self._numpy_fmts = ["%{}.{}".format(width_col(col), self.n_float) + "g"
                            for col in self.data.columns]
        self._header_formatter = [
            eval('lambda s, w=width_col(col): '  # pylint: disable=eval-used
                 '("{:>" + "{}".format(w) + "s}").format(s)',
                 {'width_col': width_col, 'col': col})
            for col in self.data.columns]

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

        If the weight is not specified, it is assumed to be 1.
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
                    logpriors=logpriors,  # type: ignore
                    loglikes=loglikes,  # type: ignore
                    derived=derived,  # type: ignore
                )
            except ValueError as valerr:
                # missing logpriors/loglikes if logpost is None,
                # or inconsistent sum if logpost given
                raise LoggedError(self.log, str(valerr)) from valerr
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
        self._cache[pos, self._icol[OutPar.minuslogpost]] = \
            -apply_temperature(logposterior.logpost, self.temperature)
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

    def _check_logps(self, temperature_only=False):
        """
        Checks the correct sums for the logpriors and likelihoods' chi-squared's, as well
        as the posterior temperature, which is returned.

        Raises ``LoggedError`` if the sums or the temperature of the sample are not
        consistent.
        """
        try:
            temperature = compute_temperature(
                -self["minuslogpost"], -self["minuslogprior"], -self["chi2"] * 0.5,
                check=True)
        except AssertionError as excpt:
            raise LoggedError(
                self.log, "The sample seems to have an inconsistent temperature.") \
                from excpt
        if not temperature_only:
            if not np.allclose(
                    self[OutPar.minuslogprior],
                    np.sum(
                        np.atleast_2d(
                            self[self.minuslogprior_names].to_numpy(dtype=np.float64)),
                        axis=-1,
                    )
            ):
                raise LoggedError(
                    self.log,
                    "The sum of logpriors in the sample is not consistent."
                )
            if not np.allclose(
                    self[OutPar.chi2],
                    np.sum(
                        np.atleast_2d(self[self.chi2_names].to_numpy(dtype=np.float64)),
                        axis=-1,
                    )
            ):
                raise LoggedError(
                    self.log,
                    "The sum of likelihood's chi2's in the sample is not consistent."
                )
        if np.isclose(temperature, 1):
            temperature = 1
        return temperature

    def _check_weights(
            self,
            weights: Optional[np.ndarray] = None,
            length: Optional[int] = None
    ):
        """
        Checks correct length, shape and signs of the ``weights``.

        If no weights passed, checks internal consistency.

        If ``length`` passed, checks for specific length of the weights vector.

        Raises ``LoggedError`` if the weights are badly arranged or invalid.
        """
        if weights is None:
            weights_array = self[OutPar.weight].to_numpy(dtype=np.float64)
        else:
            weights_array = np.atleast_1d(weights)
            if len(weights_array.shape) != 1:
                raise LoggedError(
                    self.log,
                    "The shape of the weights is wrong. Expected a 1d array, "
                    "but got shape %r.",
                    weights_array.shape
                )
            check_length = len(self) if length is None else length
            if len(weights_array) != check_length:
                raise LoggedError(
                    self.log,
                    "The length of the weights vector is wrong. Expected %d but got %d.",
                    check_length,
                    len(weights_array)
                )
        if np.any(weights_array < 0):
            raise LoggedError(
                self.log,
                "The weight vector contains negative elements."
            )

    @property
    def is_tempered(self) -> bool:
        """
        Whether the sample was obtained by drawing from a different-temperature
        distribution.
        """
        return self.temperature != 1

    @property
    def has_int_weights(self) -> bool:
        """
        Whether weights are integer.
        """
        weights = self[OutPar.weight]
        return np.allclose(np.round(weights), weights)

    def _detempered_weights(self):
        """Computes the detempered weights."""
        if self.temperature == 1:
            return self._data[OutPar.weight].to_numpy(dtype=np.float64)
        return (
            self._data[OutPar.weight].to_numpy(dtype=np.float64) *
            detempering_weights_factor(
                -self._data[OutPar.minuslogpost].to_numpy(dtype=np.float64),
                self.temperature
            )
        )

    def _detempered_minuslogpost(self):
        """Computes the detempered -log-posterior."""
        if self.temperature == 1:
            return self._data[OutPar.minuslogpost].to_numpy(dtype=np.float64)
        return -remove_temperature(
            -self._data[OutPar.minuslogpost].to_numpy(dtype=np.float64),
            self.temperature
        )

    def reset_temperature(self):
        """
        Drops the information about sampling temperature: ``weight`` and ``minuslogpost``
        columns will now correspond to those of a unit-temperature posterior sample.

        This cannot be undone: (e.g. recovering original integer tempered weights).
        You may want to call this method on a copy (see :func:`SampleCollection.copy`).
        """
        self._cache_dump()
        if self.temperature == 1:
            return
        self._data[OutPar.weight] = self._detempered_weights()
        self._drop_samples_null_weight()
        self._data[OutPar.minuslogpost] = self._detempered_minuslogpost()
        self.temperature = 1

    def _enlarge(self, n):
        """
        Enlarges the DataFrame by `n` rows.
        """
        self._data = pd.concat([
            self._data, pd.DataFrame(
                np.nan, columns=self._data.columns,
                index=np.arange(len(self._data), len(self._data) + n))])

    def _append(self, collection):
        """
        Append another collection.
        Internal method: does not check for consistency!
        """
        self._data = pd.concat([self.data, collection.data],
                               ignore_index=True)

    def __len__(self):
        return len(self._data) + (self._cache_last + 1)

    def __bool__(self):
        return len(self) != 0

    @property
    def n_last_out(self):
        """Index of the last point saved to the output."""
        return self._n_last_out

    @property  # type: ignore
    @ensure_cache_dumped
    def data(self):
        """Pandas' ``DataFrame`` containing the sample collection."""
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
        Direct access to the DataFrame, ensuring cache has been dumped.

        Returns views or copies as Pandas would do.
        """
        return self.data.__getitem__(*args)

    # MARKED FOR DEPRECATION IN v3.2
    @property
    def values(self) -> np.ndarray:
        """
        Returns the sample collection as a numpy array; to be deprecated in favour of
        ``Collection.to_numpy``, following Pandas.
        """
        print("*WARNING* Following Pandas, the 'Collection.values()' method will soon be "
              "deprecated in favour of 'Collection.to_numpy()'.")
        # BEHAVIOUR TO BE REPLACED BY AN ERROR
        return self.data.to_numpy(dtype=np.float64)
        # END OF DEPRECATION BLOCK

    def to_numpy(self, dtype=None, copy=False) -> np.ndarray:
        """Returns the sample collection as a numpy array."""
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
        self_copy = deepcopy(self)  # deletes logger (see HasLogger.__deepcopy___)
        self_copy.set_logger()
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

    def _weights_for_stats(
            self,
            first: Optional[int] = None,
            last: Optional[int] = None,
            weights: Optional[np.ndarray] = None,
            tempered: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Returns ``(weights, are_int)``, where weights can be used for computation of
        statistical quantities such as mean and covariance, and ``are_int`` is ``True``
        if the returned weights are known to be integer, and ``False`` if it cannot be
        guaranteed.

        If ``tempered=True`` (default ``False``) returns the weights of the tempered
        posterior ``p**(1/temperature)``.

        Custom weights can be passed with the argument ``weights``. In that case, internal
        weights are ignored, and some checks are performed on the passed ones.
        """
        if weights is not None:
            first = first or 0
            last = last or len(self)
            self._check_weights(weights, length=last - first)
            weights /= max(weights)
            return weights, np.allclose(np.round(weights), weights)
        if self.is_tempered and not tempered:
            # For sure the weights are not integer
            return self._detempered_weights()[first:last], False
        return (
            self[OutPar.weight][first:last].to_numpy(dtype=np.float64),
            self.has_int_weights
        )

    def mean(
            self,
            first: Optional[int] = None,
            last: Optional[int] = None,
            weights: Optional[np.ndarray] = None,
            derived: bool = False,
            tempered: bool = False
    ) -> np.ndarray:
        """
        Returns the (weighted) mean of the parameters in the chain,
        between `first` (default 0) and `last` (default last obtained),
        optionally including derived parameters if `derived=True` (default `False`).

        Custom weights can be passed with the argument ``weights``.

        If ``derived`` is ``True`` (default ``False``), the means of the derived
        parameters are included in the returned vector.

        If ``tempered=True`` (default ``False``) returns the mean of the tempered
        posterior ``p**(1/temperature)``.

        NB: For tempered samples, if passed ``tempered=False`` (default), detempered
        weights are computed on-the-fly. If this or any other function returning
        untempered statistical quantities of a tempered sample is expected to be called
        repeatedly, it would be more efficient to detemper the collection first with
        :func:`SampleCollection.reset_temperature`, and call these methods on the returned
        Collection.
        """
        if not self:
            raise LoggedError(self.log, "Collection is empty. Cannot compute mean.")
        weights_mean, _ = self._weights_for_stats(
            first, last, weights=weights, tempered=tempered)
        return np.average(self[list(self.sampled_params) +
                               (list(self.derived_params) if derived else [])]
                          [first:last].to_numpy(dtype=np.float64).T, weights=weights_mean,
                          axis=-1
                          )

    def cov(
            self,
            first: Optional[int] = None,
            last: Optional[int] = None,
            weights: Optional[np.ndarray] = None,
            derived: bool = False,
            tempered: bool = False
    ) -> np.ndarray:
        """
        Returns the (weighted) covariance matrix of the parameters in the chain,
        between `first` (default 0) and `last` (default last obtained),
        optionally including derived parameters if `derived=True` (default `False`).

        Custom weights can be passed with the argument ``weights``.

        If ``derived`` is ``True`` (default ``False``), the covariances of/with the
        derived parameters are included in the returned matrix.

        If ``tempered=True`` (default ``False``) returns the covariances of the tempered
        posterior ``p**(1/temperature)``.

        NB: For tempered samples, if passed ``tempered=False`` (default), detempered
        weights are computed on-the-fly. If this or any other function returning
        untempered statistical quantities of a tempered sample is expected to be called
        repeatedly, it would be more efficient to detemper the collection first with
        :func:`SampleCollection.reset_temperature`, and call these methods on the returned
        Collection.
        """
        if not self:
            raise LoggedError(self.log, "Collection is empty. Cannot compute cov.")
        weights_cov, are_int = self._weights_for_stats(
            first, last, weights=weights, tempered=tempered)
        weight_type_kwarg = "fweights" if are_int else "aweights"
        return np.atleast_2d(np.cov(  # type: ignore
            self[list(self.sampled_params) +
                 (list(self.derived_params) if derived else [])][first:last].to_numpy(
                dtype=np.float64).T,
            **{weight_type_kwarg: weights_cov}))

    def _drop_samples_null_weight(self):
        """Removes from the DataFrame all samples that have 0 weight."""
        self._data = self.data[self._data.weight > 0].reset_index(drop=True)
        self._n = self._data.last_valid_index() + 1

    def reweight(self, importance_weights, check=True):
        """
        Reweights the sample with the given ``importance_weights``.

        Temperature information is dropped.

        This cannot be fully undone (e.g. recovering original integer weights).
        You may want to call this method on a copy (see :func:`SampleCollection.copy`).

        For the sake of speed, length and positivity checks on the importance weights can
        be skipped with ``check=False`` (default ``True``).
        """
        self.reset_temperature()  # includes a self._cache_dump()
        if check:
            self._check_weights(importance_weights)
        self._data[OutPar.weight] *= importance_weights
        self._drop_samples_null_weight()

    def filtered_copy(self, where) -> 'SampleCollection':
        """Returns a copy of the collection with some condition ``where`` imposed."""
        return self._copy(self.data[where].reset_index(drop=True))

    def skip_samples(self, skip: float, inplace: bool = False) -> 'SampleCollection':
        """
        Skips some initial samples, or an initial fraction of them.

        For collections coming from a Nested Sampler, prints a warning and does nothing.

        Parameters
        ----------
        skip: float
            Specified the amount of initial samples to be skipped, either directly if
            ``skip>1`` (rounded up to next integer), or as a fraction if ``0<skip<1``.

        inplace: bool, default: False
            If True, returns a copy of the collection.

        Returns
        -------
        SampleCollection
            The original collection with skipped initial samples (``inplace=True``) or
            a copy of it (``inplace=False``).

        Raises
        ------
        LoggedError
            If badly defined ``skip`` value.
        """
        if skip == 0:
            return self if inplace else self.copy()
        if self.sample_type == "nested":
            self.log.warning(
                "Cannot skip initial samples from Nested Sampling samples. Doing nothing"
            )
            return self if inplace else self.copy()
        if not isinstance(skip, Number) or skip < 0:
            raise LoggedError(
                self.log,
                "Number of fraction of skipped initial samples must be positive. Got %r",
                skip
            )
        if 0 < skip < 1:
            skip = int(np.round(skip * len(self)))
        skip = int(np.round(skip))
        if inplace:
            self._data = self.data[skip:].reset_index(drop=True)
            self._n = self._data.last_valid_index() + 1
            return self
        else:
            return self.filtered_copy(slice(skip, None))

    def thin_samples(self, thin: int, inplace: bool = False) -> 'SampleCollection':
        """
        Thins the sample collection by some factor ``thin>1``.

        Parameters
        ----------
        thin: int
            Thin factor, must be ``>1``.
        inplace: bool, default: False
            If True, returns a copy of the collection.

        Returns
        -------
        SampleCollection
            Thinned version of the original collection (``inplace=True``) or a copy of it
            (``inplace=False``).

        Raises
        ------
        LoggedError
            If badly defined ``thin`` value.
        """
        if thin == 1:
            return self if inplace else self.copy()
        if thin != int(thin) or thin < 1:
            raise LoggedError(self.log, "Thin factor must be a positive integer, got %s",
                              thin)
        thin = int(thin)
        try:
            if hasattr(WeightedSamples, "thin_indices_and_weights"):
                unique, counts = \
                    WeightedSamples.thin_indices_and_weights(
                        thin, self[OutPar.weight].to_numpy(dtype=np.float64))
            else:
                raise LoggedError(self.log, "Thinning requires GetDist 1.2+", )
        except WeightedSampleError as e:
            raise LoggedError(self.log, "Error thinning: %s", e) from e
        else:
            data = self._data.iloc[unique, :].copy()
            # Produces pandas warning (pandas<2.0). Safe to ignore. Delete later.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
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
        return self.data.loc[self.data[OutPar.chi2].astype(np.float64).idxmin()].copy()

    def MAP(self):
        """Maximum-a-posteriori (MAP) sample. Returns a copy."""
        return self.data.loc[self.data[OutPar.minuslogpost].astype(np.float64).idxmin()]\
                        .copy()

    def _sampled_to_getdist(
            self,
            first: Optional[int] = None,
            last: Optional[int] = None,
            tempered: bool = False
    ) -> MCSamples:
        """
        Barebones interface with getdist. Internal use only!
        Use :func:`SampleCollection.to_getdist` instead.
        """
        names = list(self.sampled_params)
        weights, _ = self._weights_for_stats(first, last, tempered=tempered)
        if self.is_tempered and not tempered:
            minuslogposts = self._detempered_minuslogpost()[first:last]
        else:
            minuslogposts = self.data[OutPar.minuslogpost].to_numpy(
                dtype=np.float64)[first:last]
        # No logging of warnings temporarily, so getdist won't complain unnecessarily
        with NoLogging():
            mcsamples = MCSamples(
                samples=self.data[names].to_numpy(dtype=np.float64)[first:last],
                weights=weights, loglikes=minuslogposts, names=names,
            )
        return mcsamples

    def to_getdist(
            self,
            label: Optional[str] = None,
            model: Optional[Model] = None,
    ) -> MCSamples:
        """
        Parameters
        ----------
        label: str, optional
            Legend label in ``GetDist`` plots (``name_tag`` in ``GetDist`` parlance).
        model: :class:`cobaya.model.Model`, optional
            `Model` with which the sample was created. Needed only if parameter labels or
            aliases have changed since the collection was generated.

        Returns
        -------
        :class:`getdist.MCSamples`
            This collection's equivalent :class:`getdist.MCSamples` object.

        Raises
        ------
        LoggedError
            Errors when processing the arguments.
        """
        if isinstance(model, Model):
            self._cache_aux_model_quantities(model)
        elif model is not None:
            LoggedError("Optional argument `model` must be a Cobaya Model instance.")
        used_names_dict = {p: p + ("*" if p not in self.sampled_params else "")
                           for p in self.data.columns[2:]}
        return MCSamples(
            samples=self[self.data.columns[2:]].to_numpy(np.float64, copy=True),
            weights=self[OutPar.weight].to_numpy(np.float64, copy=True),
            loglikes=self[OutPar.minuslogpost].to_numpy(np.float64, copy=True),
            temperature=self.temperature,
            sampler=deepcopy(self.sample_type),
            names=list(used_names_dict.values()),
            labels=[deepcopy(self._cached_labels[p]) for p in used_names_dict
                    if p in self._cached_labels],
            ranges=deepcopy(self._cached_ranges),
            renames=deepcopy(self._cached_renames),
            name_tag=label,
            label=deepcopy(self.name),
            # ini=ini,
            # settings=settings
        )

    # Saving and updating
    def _get_driver(self, method):
        return getattr(self, method + derived_par_name_separator + self.driver)

    # Load a pre-existing file
    def _out_load(self, **kwargs):
        self._get_driver("_load")(**kwargs)
        self._cache_reset()

    # Dump/update/delete collection

    def out_update(self):
        """Update the output file to the current state of the Collection."""
        self._get_driver("_update")()

    def _out_delete(self):
        self._get_driver("_delete")()

    # txt driver
    def _load__txt(self, skip=0):
        self.log.debug("Skipping %d rows", skip)
        if bool(skip) and self.sample_type == "nested":
            raise LoggedError(
                self.log,
                "Cannot skip samples from a sample of a nested sampler. "
                "Would lead to a non-fair sample."
            )
        self._data = load_DataFrame(self.file_name, skip=skip,
                                    root_file_name=self.root_file_name)
        self.log.info("Loaded %d sample points from '%s'", len(self._data),
                      self.file_name)

    def _dump__txt(self):
        self._dump_slice__txt(0, len(self))

    def _update__txt(self):
        self._dump_slice__txt(self.n_last_out, len(self))

    def _dump_slice__txt(self, n_min=None, n_max=None):
        if n_min is None or n_max is None:
            raise LoggedError(self.log, "Needs to specify the limit n's to dump.")
        if self._n_last_out == n_max:
            return
        self._n_last_out = n_max
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
    """
    Wrapper to hold a single point, e.g. the current point of an MCMC.
    Alternative to :class:`~collection.OnePoint`, faster but with less functionality.

    For tempered samples, stores the weight and -logp of the tempered posterior (but
    untempered priors and likelihoods).
    """

    results: LogPosterior
    values: np.ndarray
    weight: int

    def __init__(self, model, temperature=1, output_thin=1):
        self.sampled_params = list(model.parameterization.sampled_params())
        self.temperature = temperature
        self.output_thin = output_thin
        self._added_weight = 0

    def add(self, values, logpost: LogPosterior):
        """Add/override the sampled point."""
        self.values = values
        if not isinstance(logpost, LogPosterior):
            raise ValueError("`logpost` argument must be LogPosterior instance.")
        self.results = logpost
        self.weight = 1

    @property
    def logpost(self):
        """:class:`~model.Logposterior` instance of the sampled point."""
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
    """
    Wrapper of :class:`~collection.SampleCollection` to hold a single point,
    e.g. the best-fit point of a minimization run (not used by default MCMC).
    """

    def __getitem__(self, columns, *args):
        if isinstance(columns, str):
            return self.data.values[0, self.data.columns.get_loc(columns)]
        try:
            return self.data.values[0,
                                    [self.data.columns.get_loc(c) for c in columns]]
        except KeyError as excpt:
            raise ValueError("Some of the indices are not valid columns.") from excpt

    def add(self, *args, **kwargs):
        """Add/override the sampled point."""
        self.reset()  # resets the DataFrame, so never goes beyond 1 point
        super().add(*args, **kwargs)

    def increase_weight(self, increase=1):
        """Increase the weight of the point by ``increase`` (default: 1)."""
        # For some reason, faster than `self.data[Par.weight] += increase`
        self.data.at[0, OutPar.weight] += increase

    # Restore original __repr__ (here, there is only 1 sample)
    def __repr__(self):
        return self.data.__repr__()

    # Dumper changed: always force to print the single element.
    def _update__txt(self):
        self._dump_slice__txt(0, 1)
