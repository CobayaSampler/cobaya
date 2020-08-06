"""
.. module:: output

:Synopsis: Generic output class and output drivers
:Author: Jesus Torrado

"""
# Global
import os
import sys
import traceback
import datetime
import re
import shutil
import platform
from packaging import version

# Local
from cobaya import __version__
from cobaya.yaml import yaml_dump, yaml_load, yaml_load_file, OutputError
from cobaya.conventions import _input_suffix, _updated_suffix, _separator_files, _version
from cobaya.conventions import _resume, _resume_default, _force, _yaml_extensions
from cobaya.conventions import _output_prefix, _debug, kinds, _params
from cobaya.log import LoggedError, HasLogger
from cobaya.input import is_equal_info, get_class
from cobaya.mpi import is_main_process, more_than_one_process, share_mpi
from cobaya.collection import Collection
from cobaya.tools import deepcopy_where_possible, find_with_regexp, sort_cosmetic

# Default output type and extension
_kind = "txt"
_ext = "txt"


def split_prefix(prefix):
    """
    Splits an output prefix into folder and file name prefix.

    If on Windows, allows for unix-like input.
    """
    if platform.system() == "Windows":
        prefix = prefix.replace("/", os.sep)
    folder = os.path.dirname(prefix) or "."
    file_prefix = os.path.basename(prefix)
    if file_prefix == ".":
        file_prefix = ""
    return folder, file_prefix


def get_info_path(folder, prefix, infix=None, kind="updated"):
    """
    Gets path to info files saved by Output.
    """
    if infix is None:
        infix = ""
    elif not infix.endswith("."):
        infix += "."
    info_file_prefix = os.path.join(
        folder, prefix + (_separator_files if prefix else ""))
    try:
        suffix = {"input": _input_suffix, "updated": _updated_suffix}[kind.lower()]
    except KeyError:
        raise ValueError("`kind` must be `input|updated`")
    return info_file_prefix + infix + suffix + _yaml_extensions[0]


class Output(HasLogger):
    """
    Basic output driver. It takes care of creating the output files, checking
    compatibility with old runs when resuming, cleaning up when forcing, preparing
    :class:`~collection.Collection` files, etc.
    """

    def __init__(self, prefix, resume=_resume_default, force=False, infix=None,
                 output_prefix=None):
        # MARKED FOR DEPRECATION IN v3.0
        # -- also remove output_prefix kwarg above
        if output_prefix is not None:
            self.log.warning("*DEPRECATION*: `output_prefix` will be deprecated in the "
                             "next version. Please use `prefix` instead.")
            # BEHAVIOUR TO BE REPLACED BY ERROR:
            prefix = output_prefix
        # END OF DEPRECATION BLOCK
        self.name = "output"  # so that the MPI-wrapped class conserves the name
        self.set_logger(self.name)
        self.folder, self.prefix = split_prefix(prefix)
        self.prefix_regexp_str = re.escape(self.prefix) + (r"\." if self.prefix else "")
        self.force = force
        if resume and force and prefix:
            # No resume and force at the same time (if output)
            raise LoggedError(
                self.log,
                "Make 'resume: True' or 'force: True', not both at the same time: "
                "can't simultaneously overwrite a chain and resume from it.")
        if not os.path.exists(self.folder):
            self.log.debug("Creating output folder '%s'", self.folder)
            try:
                os.makedirs(self.folder)
            except OSError:
                self.log.error("".join(["-"] * 20 + ["\n\n"] +
                                       list(traceback.format_exception(*sys.exc_info())) +
                                       ["\n"] + ["-"] * 37))
                raise LoggedError(
                    self.log, "Could not create folder '%s'. "
                              "See traceback on top of this message.", self.folder)
        self.log.info("Output to be read-from/written-into folder '%s', with prefix '%s'",
                      self.folder, self.prefix)
        # Prepare file names, and check if chain exists
        self.file_input = get_info_path(
            self.folder, self.prefix, infix=infix, kind="input")
        self.file_updated = get_info_path(
            self.folder, self.prefix, infix=infix, kind="updated")
        self._resuming = False
        # Output kind and collection extension
        self.kind = _kind
        self.ext = _ext
        if os.path.isfile(self.file_updated):
            self.log.info(
                "Found existing info files with the requested output prefix: '%s'",
                prefix)
            if self.force:
                self.log.info("Will delete previous products ('force' was requested).")
                self.delete_infos()
                # Sampler products will be deleted at sampler initialisation
            elif resume:
                # Only in this case we can be sure that we are actually resuming
                self._resuming = True
                self.log.info("Let's try to resume/load.")

    def is_prefix_folder(self):
        """
        Returns `True` if the output prefix is a bare folder, e.g. `chains/`.
        """
        return bool(self.prefix)

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
        return os.path.join(self.folder,
                            self.prefix + self.separator_if_needed(separator) + suffix)

    def create_folder(self, folder):
        """
        Creates the given folder (MPI-aware).
        """
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except Exception as e:
            raise LoggedError(
                self.log, "Could not create folder %r. Reason: %r", folder, str(e))

    def delete_infos(self):
        for f in [self.file_input, self.file_updated]:
            if os.path.exists(f):
                os.remove(f)

    def updated_prefix(self):
        """
        Updated path: drops folder: now it's relative to the chain's location.
        """
        return self.prefix or "."

    def is_forcing(self):
        return self.force

    def is_resuming(self):
        return self._resuming

    def set_resuming(self, value):
        self._resuming = value

    def reload_updated_info(self, cache=False, use_cache=False):
        if use_cache and getattr(self, "_old_updated_info", None):
            return self._old_updated_info
        try:
            loaded = yaml_load_file(self.file_updated)
            if cache:
                self._old_updated_info = loaded
            return deepcopy_where_possible(loaded)
        except IOError:
            if cache:
                self._old_updated_info = None
            return None

    def check_and_dump_info(self, input_info, updated_info, check_compatible=True,
                            cache_old=False, use_cache_old=False, ignore_blocks=()):
        """
        Saves the info in the chain folder twice:
           - the input info.
           - idem, populated with the components' defaults.

        If resuming a sample, checks first that old and new infos and versions are
        consistent.
        """
        # trim known params of each likelihood: for internal use only
        updated_info_trimmed = deepcopy_where_possible(updated_info)
        updated_info_trimmed[_version] = __version__
        for like_info in updated_info_trimmed.get(kinds.likelihood, {}).values():
            (like_info or {}).pop(_params, None)
        if check_compatible:
            # We will test the old info against the dumped+loaded new info.
            # This is because we can't actually check if python objects do change
            old_info = self.reload_updated_info(cache=cache_old, use_cache=use_cache_old)
            if old_info:
                new_info = yaml_load(yaml_dump(updated_info_trimmed))
                if not is_equal_info(old_info, new_info, strict=False,
                                     ignore_blocks=list(ignore_blocks) + [
                                         _output_prefix]):
                    raise LoggedError(
                        self.log, "Old and new run information not compatible! "
                                  "Resuming not possible!")
                # Deal with version comparison separately:
                # - If not specified now, take the one used in resume info
                # - If specified both now and before, check new older than old one
                # (For Cobaya's own version, prefer new one always)
                old_version = old_info.get(_version, None)
                new_version = new_info.get(_version, None)
                if old_version:
                    if version.parse(old_version) > version.parse(new_version):
                        raise LoggedError(
                            self.log, "You are trying to resume a run performed with a "
                                      "newer version of Cobaya: %r (you are using %r). "
                                      "Please, update your Cobaya installation.",
                            old_version, new_version)
                for k in (kind for kind in kinds if kind in updated_info):
                    if k in ignore_blocks:
                        continue
                    for c in updated_info[k]:
                        new_version = updated_info[k][c].get(_version)
                        old_version = old_info[k][c].get(_version)
                        if new_version is None:
                            updated_info[k][c][_version] = old_version
                            updated_info_trimmed[k][c][_version] = old_version
                        elif old_version is not None:
                            cls = get_class(c, k, None_if_not_found=True)
                            if cls and cls.compare_versions(
                                    old_version, new_version, equal=False):
                                raise LoggedError(
                                    self.log, "You have requested version %r for "
                                              "%s:%s, but you are trying to resume a "
                                              "run that used a newer version: %r.",
                                    new_version, k, c, old_version)
        # If resuming, we don't want to to *partial* dumps
        if ignore_blocks and self.is_resuming():
            return
        # Work on a copy of the input info, since we are updating the prefix
        # (the updated one is already a copy)
        if input_info is not None:
            input_info = deepcopy_where_possible(input_info)
        # Write the new one
        for f, info in [(self.file_input, input_info),
                        (self.file_updated, updated_info_trimmed)]:
            if info:
                for k in ignore_blocks:
                    info.pop(k, None)
                info.pop(_debug, None)
                info.pop(_force, None)
                info.pop(_resume, None)
                # make sure the dumped output_prefix does only contain the file prefix,
                # not the folder, since it's already been placed inside it
                info[_output_prefix] = self.updated_prefix()
                with open(f, "w", encoding="utf-8") as f_out:
                    try:
                        f_out.write(yaml_dump(sort_cosmetic(info)))
                    except OutputError as e:
                        raise LoggedError(self.log, str(e))

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
                    "From regexp %r in folder %r, deleting files %r", regexp.pattern,
                    root, file_names)
        else:
            file_names = [root]
            self.log.debug("Deleting folder %r", root)
        for f in file_names:
            try:
                os.remove(f)
            except IsADirectoryError:
                try:
                    shutil.rmtree(f)
                except:
                    raise
            except OSError:
                pass

    def prepare_collection(self, name=None, extension=None):
        """
        Generates a file name for the collection, as
        ``[folder]/[prefix].[name].[extension]``.

        Notice that ``name=None`` generates a date, but ``name=""`` removes the ``name``
        field, making it simply ``[folder]/[prefix].[extension]``.
        """
        if name is None:
            name = (datetime.datetime.now().isoformat().replace("T", "")
                        .replace(":", "").replace(".", "")
                        .replace("-", "")[:(4 + 2 + 2) + (2 + 2 + 2 + 3)])  # up to ms
        file_name = os.path.join(
            self.folder,
            self.prefix + ("." if self.prefix else "") + (name + "." if name else "") +
            self.sanitize_collection_extension(extension))
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
        return re.compile(self.prefix_regexp_str + name +
                          re.escape(extension.lower()) + "$")

    def is_collection_file_name(self, file_name, name=None, extension=None):
        """
        Check if a `file_name` is a collection compatible with this `Output` instance.

        Use `name` for particular types of collections (default: any number).
        Pass `False` to mean there is nothing between the output prefix and the extension.
        """
        return (file_name ==
                getattr(self.collection_regexp(name=name, extension=extension)
                        .match(file_name), "group", lambda: None)())

    def find_collections(self, name=None, extension=None):
        """
        Returns all collection files found which are compatible with this `Output`
        instance, including their path in their name.

        Use `name` for particular types of collections (default: matches any number).
        Pass `False` to mean there is nothing between the output prefix and the extension.
        """
        return [
            f2 for f2 in [os.path.join(self.folder, f) for f in os.listdir(self.folder)]
            if self.is_collection_file_name(
                os.path.split(f2)[1], name=name, extension=extension)]

    def load_collections(self, model, skip=0, thin=1, concatenate=False,
                         name=None, extension=None):
        """
        Loads all collection files found which are compatible with this `Output`
        instance, including their path in their name.

        Use `name` for particular types of collections (default: any number).
        Pass `False` to mean there is nothing between the output prefix and the extension.
        """
        filenames = self.find_collections(name=name, extension=extension)
        collections = [
            Collection(model, self, name="%d" % (1 + i), file_name=filename,
                       load=True, onload_skip=skip, onload_thin=thin)
            for i, filename in enumerate(filenames)]
        if concatenate and collections:
            collection = collections[0]
            for collection_i in collections[1:]:
                collection.append(collection_i)
            collections = collection
        return collections


class OutputDummy(Output):
    """
    Dummy output class. Does nothing. Evaluates to 'False' as a class.
    """

    def __init__(self, *args, **kwargs):
        self.set_logger(lowercase=True)
        self.log.debug("No output requested. Doing nothing.")
        # override all methods that actually produce output
        exclude = ["nullfunc"]
        _func_name = "__name__"
        for attrname, attr in list(Output.__dict__.items()):
            func_name = getattr(attr, _func_name, None)
            if func_name and func_name not in exclude and '__' not in func_name:
                setattr(self, attrname, self.nullfunc)

    def nullfunc(self, *args, **kwargs):
        pass

    def __nonzero__(self):
        return False

    def __bool__(self):
        return False


class Output_MPI(Output):
    """
    MPI wrapper around the Output class. Makes sure actual I/O operations are only done
    once (except the opposite is explicitly requested).
    """

    def __init__(self, *args, **kwargs):
        if is_main_process():
            Output.__init__(self, *args, **kwargs)
        if more_than_one_process():
            to_broadcast = (
                "folder", "prefix", "kind", "ext", "_resuming", "prefix_regexp_str")
            values = share_mpi([getattr(self, var) for var in to_broadcast]
                               if is_main_process() else None)
            for name, var in zip(to_broadcast, values):
                setattr(self, name, var)

    def check_and_dump_info(self, *args, **kwargs):
        if is_main_process():
            Output.check_and_dump_info(self, *args, **kwargs)
        # Share cached loaded info
        self._old_updated_info = share_mpi(getattr(self, "_old_updated_info", None))

    def reload_updated_info(self, *args, **kwargs):
        if is_main_process():
            return Output.reload_updated_info(self, *args, **kwargs)
        else:
            # Only cached possible when non main process
            if not kwargs.get("use_cache"):
                raise ValueError(
                    "Cannot call `reload_updated_info` from non-main process "
                    "unless cached version (`use_cache=True`) requested.")
            return self._old_updated_info

    def create_folder(self, *args, **kwargs):
        if is_main_process():
            Output.create_folder(self, *args, **kwargs)

    def set_resuming(self, *args, **kwargs):
        if is_main_process():
            Output.set_resuming(self, *args, **kwargs)
        if more_than_one_process():
            self._resuming = share_mpi(self._resuming if is_main_process() else None)


def get_output(*args, **kwargs):
    """
    Auxiliary function to retrieve the output driver
    (e.g. whether to get the MPI-wrapped one, or a dummy output driver).
    """
    # MARKED FOR DEPRECATION IN v3.0
    if kwargs.get("output_prefix") is not None:
        kwargs["prefix"] = kwargs["output_prefix"]
    # END OF DEPRECATION BLOCK
    if kwargs.get("prefix"):
        from cobaya.mpi import import_MPI
        return import_MPI(".output", "Output")(*args, **kwargs)
    else:
        return OutputDummy(*args, **kwargs)
