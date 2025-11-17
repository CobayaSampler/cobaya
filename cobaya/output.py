"""
.. module:: output

:Synopsis: Generic output class and output drivers
:Author: Jesus Torrado

"""

import datetime
import importlib
import os
import re
import shutil
import sys
import time
from typing import Any

from packaging import version

from cobaya import mpi
from cobaya.component import get_component_class
from cobaya.conventions import Extension, get_version, kinds, resume_default
from cobaya.input import get_info_path, is_equal_info, load_info_dump, split_prefix
from cobaya.log import HasLogger, LoggedError, get_logger, get_traceback_text
from cobaya.tools import (
    deepcopy_where_possible,
    find_with_regexp,
    has_non_yaml_reproducible,
    sort_cosmetic,
)
from cobaya.typing import InputDict
from cobaya.yaml import (
    InputImportError,
    OutputError,
    yaml_dump,
    yaml_load,
    yaml_load_file,
)

# Default output type and extension
_kind = "txt"
_ext = "txt"


def use_portalocker():
    if os.getenv("COBAYA_USE_FILE_LOCKING", "t").lower() in ("true", "1", "t"):
        if importlib.util.find_spec("portalocker") is not None:
            return True
        else:
            return None
    return False


class FileLock:
    _file_handle: Any

    def __init__(self, filename=None, log=None, **kwargs):
        self.lock_error_file = ""
        self.lock_file = ""
        if filename:
            self.set_lock(log, filename, **kwargs)
        else:
            assert not log and not kwargs

    def set_lock(self, log, filename, force=False, wait=False):
        if self.has_lock():
            return
        self.lock_file = filename + ".locked"
        self.lock_error_file = filename + ".lock_err"
        try:
            os.remove(self.lock_error_file)
        except OSError:
            pass
        self.log = log or get_logger("file_lock")
        try:
            h: Any = None
            if use_portalocker():
                import portalocker

                try:
                    h = open(self.lock_file, "wb")
                    flags = portalocker.LOCK_EX
                    if not wait:
                        flags |= portalocker.LOCK_NB
                    portalocker.lock(h, flags)
                    self._file_handle = h
                except portalocker.exceptions.BaseLockException:
                    if h:
                        h.close()
                    self.lock_error()
            else:
                # will work, but crashes will leave .lock files that will raise error
                if wait:
                    while True:
                        try:
                            self._file_handle = open(
                                self.lock_file, "wb" if force else "xb"
                            )
                            break
                        except FileExistsError:
                            # Wait for other process to clear lock or report error
                            if os.path.exists(self.lock_error_file):
                                self.lock_error()
                            time.sleep(0.1)
                        except OSError:
                            self.lock_error()
                else:
                    self._file_handle = open(self.lock_file, "wb" if force else "xb")
        except OSError:
            self.lock_error()

    def lock_error(self):
        if not self.has_lock():
            assert self.lock_error_file
            try:
                # make lock_err so process holding lock can check
                # another process had an error
                with open(self.lock_error_file, "wb"):
                    pass
            except OSError:
                pass
        if mpi.is_disabled():
            raise LoggedError(
                self.log,
                "File %s is locked by another process, you are running "
                "with MPI disabled but may have more than one process. "
                "Make sure that you have mpi4py installed and working."
                "Note that --test should not be used with MPI.",
                self.lock_file,
            )
        if mpi.get_mpi():
            import mpi4py
        else:
            mpi4py = None
        if mpi.is_main_process() and use_portalocker() is None:
            self.log.warning('install "portalocker" for better file lock control.')
        raise LoggedError(
            self.log,
            "File %s is locked.\nYou may be running multiple jobs with "
            "the same output when you intended to run with MPI. "
            "Check that mpi4py is correctly installed and "
            "configured (using the same mpi as mpirun/mpiexec); "
            "e.g. try the test at\n"
            "https://cobaya.readthedocs.io/en/latest/installation."
            "html#mpi-parallelization-optional-but-encouraged\n"
            + (
                "Your current mpi4py config is:\n %s" % mpi4py.get_config()
                if mpi4py is not None
                else "mpi4py is NOT currently installed."
            )
            + "\nIf this is a lock issue you can disable this check by "
            "setting env COBAYA_USE_FILE_LOCKING=False.",
            self.lock_file,
        )

    def check_error(self):
        if self.lock_error_file and os.path.exists(self.lock_error_file):
            self.lock_error()

    def clear_lock(self):
        if self.has_lock():
            self._file_handle.close()
            del self._file_handle
            try:
                os.remove(self.lock_file)
            except OSError:
                pass
            try:
                os.remove(self.lock_error_file)
            except OSError:
                pass
        self.lock_error_file = ""
        self.lock_file = ""

    def has_lock(self):
        return hasattr(self, "_file_handle")

    def __del__(self):
        self.clear_lock()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # Always clear any held lock; propagate original exceptions.
        self.clear_lock()
        return False


class OutputReadOnly:
    """
    A read-only output driver: it tracks naming of, and can load input and collection
    files. Contrary to :class:`output.Output`, this class is not MPI-aware, which makes it
    useful to be able to do these operations within isolated MPI processes.
    """

    _old_updated_info: InputDict | None

    def __init__(self, prefix, infix=None):
        self.folder, self.prefix = split_prefix(prefix)
        self.prefix_regexp_str = re.escape(self.prefix) + (
            r"[\._]" if self.prefix else ""
        )
        # Prepare file names, and check if chain exists
        self.file_input = get_info_path(
            self.folder, self.prefix, infix=infix, kind="input"
        )
        self.file_updated = get_info_path(self.folder, self.prefix, infix=infix)
        self.dump_file_updated = get_info_path(
            self.folder, self.prefix, infix=infix, ext=Extension.dill
        )
        # Output kind and collection extension
        self.kind = _kind
        self.ext = _ext

    def __str__(self):
        return (
            f"Output instance defined within folder '{self.folder}' "
            f"with prefix '{self.prefix}'."
        )

    def __repr__(self):
        return self.__str__()

    def is_prefix_folder(self):
        """
        Returns `True` if the output prefix is a bare folder, e.g. `chains/`.
        """
        return bool(self.prefix)

    def updated_prefix(self):
        """
        Updated path: drops folder: now it's relative to the chain's location.
        """
        return self.prefix or "."

    def separator_if_needed(self, separator):
        """
        Returns the given separator if there is an actual file name prefix (i.e. the
        output prefix is not a bare folder), or an empty string otherwise.

        Useful to add custom suffixes to output prefixes (may want to use
        `Output.add_suffix` for that).
        """
        return separator if self.is_prefix_folder() else ""

    def sanitize_collection_extension(self, extension):
        """
        Returns the `extension` without the leading dot, if given, or the default one
        `Output.ext` otherwise.
        """
        if extension:
            return extension.lstrip(".")
        return self.ext

    def add_suffix(self, suffix, separator="_"):
        """
        Returns the full output prefix (folder and file name prefix) combined with a
        given suffix, inserting a given separator in between (default: `_`) if needed.
        """
        return os.path.join(
            self.folder, self.prefix + self.separator_if_needed(separator) + suffix
        )

    def get_updated_info(self, use_cache=False, cache=False) -> InputDict | None:
        """
        Returns the version of the input file updated with defaults, loading it if
        necessary not previously cached, or if forced by ``use_cache=False``.

        If loading is forced and ``cache=True``, the loaded input will be cached for
        future calls.
        """
        if use_cache and hasattr(self, "_old_updated_info"):
            return self._old_updated_info
        return self.reload_updated_info(cache=cache)

    def reload_updated_info(self, cache=False) -> InputDict | None:
        """
        Reloads and returns the version of the input file updated with defaults.

        If none is found, returns ``None`` without raising an error.

        If ``cache=True``, the loaded input will be cached for future calls.
        """
        try:
            if os.path.isfile(self.dump_file_updated):
                loaded = load_info_dump(self.dump_file_updated)
            else:
                loaded = yaml_load_file(self.file_updated)  # type: ignore
            if cache:
                self._old_updated_info = deepcopy_where_possible(loaded)
            return loaded  # type: ignore
        except OSError:
            if cache:
                self._old_updated_info = None
            return None

    def prepare_collection(self, name=None, extension=None):
        """
        Generates a file name for the collection, as
        ``[folder]/[prefix].[name].[extension]``.

        Notice that ``name=None`` generates a date, but ``name=""`` removes the ``name``
        field, making it simply ``[folder]/[prefix].[extension]``.
        """
        if name is None:
            name = (
                datetime.datetime.now()
                .isoformat()
                .replace("T", "")
                .replace(":", "")
                .replace(".", "")
                .replace("-", "")[: (4 + 2 + 2) + (2 + 2 + 2 + 3)]
            )  # up to ms
        file_name = os.path.join(
            self.folder,
            self.prefix
            + ("." if self.prefix else "")
            + (name + "." if name else "")
            + self.sanitize_collection_extension(extension),
        )
        return file_name, self.kind

    def collection_regexp(self, name=None, extension=None):
        """
        Returns a regexp for collections compatible with this output settings.

        Use `name` for particular types of collections (default: any number).
        Pass `False` to mean there is nothing between the output prefix and the extension.
        """
        if name is None:
            name = r"\d+\."
        elif name is False:
            name = ""
        else:
            name = re.escape(name) + r"\."
        extension = self.sanitize_collection_extension(extension)
        return re.compile(
            self.prefix_regexp_str + name + re.escape(extension.lower()) + "$"
        )

    def is_collection_file_name(self, file_name, name=None, extension=None):
        """
        Check if a `file_name` is a collection compatible with this `Output` instance.

        Use `name` for particular types of collections (default: any number).
        Pass `False` to mean there is nothing between the output prefix and the extension.
        """
        return (
            file_name
            == getattr(
                self.collection_regexp(name=name, extension=extension).match(file_name),
                "group",
                lambda: None,
            )()
        )

    def find_collections(self, name=None, extension=None):
        """
        Returns all collection files found which are compatible with this `Output`
        instance, including their path in their name.

        Use `name` for particular types of collections (default: matches any number).
        Pass `False` to mean there is nothing between the output prefix and the extension.
        """
        return sorted(
            f2
            for f2 in [os.path.join(self.folder, f) for f in os.listdir(self.folder)]
            if self.is_collection_file_name(
                os.path.split(f2)[1], name=name, extension=extension
            )
        )

    def load_collections(
        self,
        model,
        skip=0,
        thin=1,
        combined=False,
        name=None,
        extension=None,
        check_logp_sums=True,
    ):
        """
        Loads all collection files found which are compatible with this `Output`
        instance, including their path in their name.

        Use `name` for particular types of collections (default: any number).
        Pass `False` to mean there is nothing between the output prefix and the extension.

        If ``check_logp_sums=False`` allows for samples to have individual chi2's and
        logpriors that do not add up to the total ones, or are undefined.

        Notes
        -----
        Unless you know what you are doing, use the :func:`cobaya.output.load_samples`
        function instead to load samples.
        """
        filenames = self.find_collections(name=name, extension=extension)
        from cobaya.collection import SampleCollection

        collections = [
            SampleCollection(
                model,
                self,
                name="%d" % (1 + i),
                file_name=filename,
                load=True,
                onload_skip=skip,
                onload_thin=thin,
                is_batch=len(filenames) > 1,
                check_logp_sums=check_logp_sums,
            )
            for i, filename in enumerate(filenames)
        ]
        if combined and collections:
            collection = collections[0]
            for collection_i in collections[1:]:
                collection._append(collection_i)
            collection.is_batch = False
            return collection
        return collections


class Output(HasLogger, OutputReadOnly):
    """
    Basic output driver. It takes care of creating the output files, checking
    compatibility with old runs when resuming, cleaning up when forcing, preparing
    :class:`~collection.SampleCollection` files, etc.
    """

    @mpi.set_from_root(
        (
            "force",
            "folder",
            "prefix",
            "kind",
            "ext",
            "file_input",
            "file_updated",
            "dump_file_updated",
            "_resuming",
            "prefix_regexp_str",
            "log",
        )
    )
    def __init__(self, prefix, resume=resume_default, force=False, infix=None):
        OutputReadOnly.__init__(self, prefix, infix)
        self.name = "output"
        self.set_logger(self.name)
        self.lock = FileLock()
        self.force = force
        if resume and force and prefix and infix != "minimize":
            # No resume and force at the same time (if output)
            raise LoggedError(
                self.log,
                "Make 'resume: True' or 'force: True', not both at the same time: "
                "can't simultaneously overwrite a chain and resume from it.",
            )
        if not os.path.exists(self.folder):
            self.log.debug("Creating output folder '%s'", self.folder)
            try:
                os.makedirs(self.folder)
            except OSError as excpt:
                self.log.error(get_traceback_text(sys.exc_info()))
                raise LoggedError(
                    self.log,
                    "Could not create folder '%s'. See traceback on top of this message.",
                    self.folder,
                ) from excpt
        self.log.info(
            "Output to be read-from/written-into folder '%s', with prefix '%s'",
            self.folder,
            self.prefix,
        )
        self._resuming = False
        self._has_old_updated_info = os.path.isfile(self.file_updated)
        if self._has_old_updated_info:
            self.log.info(
                "Found existing info files with the requested output prefix: '%s'", prefix
            )
            if self.force:
                self.log.info("Will delete previous products ('force' was requested).")
                self.delete_infos()
                # Sampler products will be deleted at sampler initialisation
            elif resume:
                # Only in this case we can be sure that we are actually resuming
                self._resuming = True
                self.log.info("Let's try to resume/load.")
            else:
                self.log.debug(
                    "There was old updated info, but no resume or force requested. "
                    "Behavior will be handled by sampler."
                )

    @mpi.root_only
    def create_folder(self, folder):
        """
        Creates the given folder (MPI-aware).
        """
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except Exception as e:
            raise LoggedError(
                self.log, "Could not create folder %r. Reason: %r", folder, str(e)
            ) from e

    @mpi.root_only
    def delete_infos(self):
        self.check_lock()
        for f in [self.file_input, self.file_updated, self.dump_file_updated]:
            try:
                os.remove(f)
            except OSError:
                pass

    def is_resuming(self):
        return self._resuming

    @mpi.set_from_root("_resuming")
    def set_resuming(self, value):
        self._resuming = value

    def reload_updated_info(self, cache=False) -> InputDict | None:
        """
        Reloads and returns the version of the input file updated with defaults.

        If none is found, returns ``None`` without raising an error.

        If ``cache=True``, the loaded input will be cached for future calls.
        """
        loaded = None
        if mpi.is_main_process():
            loaded = super().reload_updated_info(cache=cache)
        loaded = mpi.share_mpi(loaded)
        if cache:
            self._old_updated_info = loaded
        return loaded

    def check_and_dump_info(
        self,
        input_info,
        updated_info,
        check_compatible=True,
        cache_old=False,
        use_cache_old=False,
        ignore_blocks=(),
    ):
        """
        Saves the info in the chain folder twice:
           - the input info.
           - idem, populated with the components' defaults.

        If resuming a sample, checks first that old and new infos and versions are
        consistent unless allow_changes is True.
        """
        # trim known params of each likelihood: for internal use only
        self.check_lock()
        updated_info_trimmed = deepcopy_where_possible(updated_info)
        updated_info_trimmed["version"] = get_version()
        for like_info in updated_info_trimmed.get("likelihood", {}).values():
            (like_info or {}).pop("params", None)
        if check_compatible or cache_old:
            # We will test the old info against the dumped+loaded new info.
            # This is because we can't actually check if python objects do change
            try:
                old_info = self.get_updated_info(cache=cache_old, use_cache=use_cache_old)
            except InputImportError:
                # for example, when there's a dynamically generated class that cannot
                # be found by the yaml loader (could use yaml loader that ignores them)
                old_info = None
            if check_compatible and old_info and not old_info.get("test"):
                old_info = yaml_load(yaml_dump(old_info))
                new_info = yaml_load(yaml_dump(updated_info_trimmed))
                if not is_equal_info(
                    old_info,
                    new_info,
                    strict=False,
                    ignore_blocks=list(ignore_blocks) + ["output"],
                ):
                    raise LoggedError(
                        self.log,
                        "Old and new run information not compatible! "
                        "Resuming not possible!\n"
                        "Use --allow-changes to proceed anyway.",
                    )
                # Deal with version comparison separately:
                # - If not specified now, take the one used in resume info
                # - If specified both now and before, check new older than old one
                # (For Cobaya's own version, prefer new one always)
                old_version = old_info.get("version")
                new_version = new_info.get("version")
                if isinstance(old_version, str) and isinstance(new_version, str):
                    if version.parse(old_version) > version.parse(new_version):
                        raise LoggedError(
                            self.log,
                            "You are trying to resume a run performed with a "
                            "newer version of Cobaya: %r (you are using %r). "
                            "Please, update your Cobaya installation.",
                            old_version,
                            new_version,
                        )
                for k in set(kinds).intersection(updated_info):
                    if k in ignore_blocks or updated_info[k] is None:
                        continue
                    for c in updated_info[k]:
                        new_version = updated_info[k][c].get("version")
                        old_version = old_info[k][c].get("version")  # type: ignore
                        if new_version is None:
                            updated_info[k][c]["version"] = old_version
                            updated_info_trimmed[k][c]["version"] = old_version
                        elif old_version is not None:
                            cls = get_component_class(
                                c,
                                k,
                                class_name=updated_info[k][c].get("class"),
                                logger=self.log,
                            )
                            if cls and cls.compare_versions(
                                old_version, new_version, equal=False
                            ):
                                raise LoggedError(
                                    self.log,
                                    "You have requested version %r for "
                                    "%s:%s, but you are trying to resume a "
                                    "run that used a newer version: %r.",
                                    new_version,
                                    k,
                                    c,
                                    old_version,
                                )
        # If resuming, we don't want to do *partial* dumps
        if ignore_blocks and self.is_resuming():
            return
        # Work on a copy of the input info, since we are updating the prefix
        # (the updated one is already a copy)
        if input_info is not None:
            input_info = deepcopy_where_possible(input_info)
        # Write the new one
        for f, info in [
            (self.file_input, input_info),
            (self.file_updated, updated_info_trimmed),
        ]:
            if info:
                for k in tuple(ignore_blocks) + ("debug", "force", "resume"):
                    info.pop(k, None)
                # make sure the dumped output prefix does only contain the file prefix,
                # not the folder, since it's already been placed inside it
                info["output"] = self.updated_prefix()
                with open(f, "w", encoding="utf-8") as f_out:
                    try:
                        f_out.write(yaml_dump(sort_cosmetic(info)))
                    except OutputError as e:
                        raise LoggedError(self.log, str(e)) from e
        if updated_info_trimmed and has_non_yaml_reproducible(updated_info_trimmed):
            try:
                import dill
            except ImportError:
                self.mpi_info('Install "dill" to save reproducible options file.')
            else:
                import pickle

                try:
                    with open(self.dump_file_updated, "wb") as f:
                        dill.dump(
                            sort_cosmetic(updated_info_trimmed),
                            f,
                            pickle.HIGHEST_PROTOCOL,
                        )
                except pickle.PicklingError as e:
                    os.remove(self.dump_file_updated)
                    self.mpi_info("Options file cannot be pickled %s", e)

    def load_collections(
        self,
        model,
        skip=0,
        thin=1,
        combined=False,
        name=None,
        extension=None,
        concatenate=None,
    ):
        """
        Loads all collection files found which are compatible with this `Output`
        instance, including their path in their name.

        Use `name` for particular types of collections (default: any number).
        Pass `False` to mean there is nothing between the output prefix and the extension.

        Notes
        -----
        Unless you know what you are doing, use the :func:`cobaya.output.load_samples`
        function instead to load samples.
        """
        self.check_lock()
        return super().load_collections(
            model,
            skip=skip,
            thin=thin,
            combined=combined,
            name=name,
            extension=extension,
            concatenate=concatenate,
        )

    def delete_with_regexp(self, regexp, root=None):
        """
        Deletes all files compatible with the given `regexp`.

        If `regexp` is `None` and `root` is defined, deletes the `root` folder.
        """
        if root is None:
            root = self.folder
        if regexp is not None:
            file_names = find_with_regexp(regexp, root)
            if file_names:
                self.log.debug(
                    "From regexp %r in folder %r, deleting files %r",
                    regexp.pattern,
                    root,
                    file_names,
                )
        else:
            file_names = [root]
            self.log.debug("Deleting folder %r", root)
        for f in file_names:
            self.delete_file_or_folder(f)

    def delete_file_or_folder(self, filename):
        """
        Deletes a file or a folder. Fails silently.
        """
        self.check_lock()
        try:
            os.remove(filename)
        except IsADirectoryError:
            try:
                shutil.rmtree(filename)
            except Exception:
                self.log.debug("Tried and failed to delete folder %r", filename)
                raise
        except OSError:
            pass

    @mpi.root_only
    def clear_lock(self):
        self.lock.clear_lock()

    @mpi.root_only
    def check_lock(self):
        self.lock.check_error()

    @mpi.root_only
    def set_lock(self):
        self.lock.set_lock(self.log, self.file_input, force=self.force)

    def __enter__(self):
        self.set_lock()
        return self

    def __exit__(self, *args):
        self.clear_lock()


class OutputDummy(Output):
    """
    Dummy output class. Does nothing. Evaluates to 'False' as a class.
    """

    def __init__(self, *args, **kwargs):
        self.set_logger()
        self.log.debug("No output requested. Doing nothing.")
        # override all methods that actually produce output
        exclude = ["nullfunc"]
        _func_name = "__name__"
        for attrname, attr in list(Output.__dict__.items()):
            func_name = getattr(attr, _func_name, None)
            if func_name and func_name not in exclude and "__" not in func_name:
                setattr(self, attrname, self.nullfunc)

    def nullfunc(self, *args, **kwargs):
        pass

    def __str__(self):
        return "DummyOutput instance (does no do any I/O)."

    def __nonzero__(self):
        return False

    def __bool__(self):
        return False


def get_output(*args, **kwargs) -> Output:
    """
    Auxiliary function to retrieve the output driver
    (e.g. whether to get the MPI-wrapped one, or a dummy output driver).
    """
    if kwargs.get("prefix") or len(args) >= 1:
        return Output(*args, **kwargs)
    else:
        return OutputDummy(*args, **kwargs)


def load_samples(
    prefix,
    skip=0,
    thin=1,
    combined=False,
    to_getdist=False,
    check_logp_sums=True,
):
    """
    Loads all sample collections with a given prefix.

    Parameters
    ----------
    prefix: str
        Prefix used to look for samples (e.g. ``chains/a`` will try to load
        ``chains/a.1.txt`` etc.). Can also be passed a ``.yaml`` file from which the
        prefix will be guessed.
    skip: float (default: 0)
        Specifies the amount of initial samples to be skipped, either directly if
        ``skip>1`` (rounded up to next integer), or as a fraction if ``0<skip<1``.
        For collections coming from a Nested Sampler, prints a warning and does nothing.
    thin: int (default: 1, no skipping)
        Specifies a thinning factor applied to the sample; must be ``>1``.
    combined: bool (default: False)
        If True, instead of returning a list of all collections of samples found, it
        returns a single concatenated one, after applying ``skip`` and ``thin`` on the
        individual ones. NB: if ``combined`` is True, it is recommended to skip some
        initial fraction, e.g. ``skip=0.3``.
    to_getdist: bool (default: False)
        If ``True``, returns a single :class:`getdist.MCSamples` instance, containing all
        samples (``combined`` is ignored).
    check_logp_sums: bool (default: True)
        If ``False`` allows for samples to have individual chi2's and logpriors that do not
        add up to the total ones, or are undefined.

    Returns
    -------
    samples: list[SampleCollection], SampleCollection, getdist.MCSamples
        A list of :class:`~collection.SampleCollection`, or a single one if passed
        ``combined=True``. If Cobaya output was detected for the given prefix, but no
        samples were found (e.g. a failed or early-cancelled run), an empty list will be
        returned.

    Raises
    ------
    FileNotFoundError: if no Cobaya output was found.

    Notes
    -----
    This function does not interact directly with MPI, and will return the same value for
    all processes.
    """
    # yaml: load and look for "output", or use file name without extension
    is_yaml_filename = isinstance(prefix, str) and any(
        os.path.splitext(prefix)[1].lower() == ext.lower() for ext in Extension.yamls
    )
    if is_yaml_filename:
        file_name, _ = os.path.splitext(prefix)
        prefix = (yaml_load_file(prefix) or {}).get("output", None)
        if prefix is None:
            prefix = file_name
    output = OutputReadOnly(prefix=prefix)
    info = output.get_updated_info()
    if info is None:
        raise FileNotFoundError(
            f"Could not find any sample with prefix '{prefix}' "
            f"(looked for file '{output.file_updated}')."
        )
    from cobaya.model import DummyModel

    dummy_model = DummyModel(info["params"], info["likelihood"], info.get("prior"))
    if to_getdist:
        collections = output.load_collections(
            dummy_model,
            skip=skip,
            thin=thin,
            combined=False,
            check_logp_sums=check_logp_sums,
        )
        if collections:
            collections = collections[0].to_getdist(combine_with=collections[1:])
    else:
        collections = output.load_collections(
            dummy_model,
            skip=skip,
            thin=thin,
            combined=combined,
            check_logp_sums=check_logp_sums,
        )
    return collections
