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
from itertools import chain
import re

# Local
from cobaya.yaml import yaml_dump, yaml_load, yaml_load_file, OutputError
from cobaya.conventions import _input_suffix, _updated_suffix, _separator_files, _version
from cobaya.conventions import _resume, _resume_default, _force, _yaml_extensions
from cobaya.conventions import kinds, _params
from cobaya.log import LoggedError, HasLogger
from cobaya.input import is_equal_info, get_class
from cobaya.mpi import is_main_process, more_than_one_process, share_mpi
from cobaya.collection import Collection
from cobaya.tools import deepcopy_where_possible

# Default output type and extension
_kind = "txt"
_ext = "txt"


class Output(HasLogger):
    def __init__(self, output_prefix=None, resume=_resume_default, force=False):
        self.name = "output"  # so that the MPI-wrapped class conserves the name
        self.set_logger(self.name)
        self.folder = os.sep.join(output_prefix.split(os.sep)[:-1]) or "."
        self.prefix = (lambda x: x if x != "." else "")(output_prefix.split(os.sep)[-1])
        self.prefix_regexp_str = \
            os.path.join(self.folder, self.prefix) + (r"\." if self.prefix else "")
        self.force = force
        if resume and force and output_prefix:
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
        info_file_prefix = os.path.join(
            self.folder, self.prefix + (_separator_files if self.prefix else ""))
        self.file_input = info_file_prefix + _input_suffix + _yaml_extensions[0]
        self.file_updated = info_file_prefix + _updated_suffix + _yaml_extensions[0]
        self.resuming = False
        # Output kind and collection extension
        self.kind = _kind
        self.ext = _ext
        if os.path.isfile(self.file_updated):
            self.log.info(
                "Found existing info files with the requested output prefix: '%s'",
                output_prefix)
            if self.force:
                self.log.info("Will delete previous products ('force' was requested).")
                self.delete_infos()
                # Sampler products will be deleted at sampler initialisation
            elif resume:
                # Only in this case we can be sure that we are actually resuming
                self.resuming = True
                self.log.info("Let's try to resume/load.")

    def delete_infos(self):
        for f in [self.file_input, self.file_updated]:
            if os.path.exists(f):
                os.remove(f)

    def updated_output_prefix(self):
        """
        Updated path: drops folder: now it's relative to the chain's location.
        """
        return self.prefix or "."

    def is_forcing(self):
        return self.force

    def is_resuming(self):
        return self.resuming

    def reload_updated_info(self):
        return yaml_load_file(self.file_updated)

    def dump_info(self, input_info, updated_info, check_compatible=True):
        """
        Saves the info in the chain folder twice:
           - the input info.
           - idem, populated with the modules' defaults.

        If resuming a sample, checks first that old and new infos and versions are
        consistent.
        """
        # trim known params of each likelihood: for internal use only
        updated_info_trimmed = deepcopy_where_possible(updated_info)
        for lik_info in updated_info_trimmed.get(kinds.likelihood, {}).values():
            if hasattr(lik_info, "pop"):
                lik_info.pop(_params, None)
        if check_compatible:
            try:
                # We will test the old info against the dumped+loaded new info.
                # This is because we can't actually check if python objects do change
                old_info = self.reload_updated_info()
                if not old_info:
                    raise LoggedError(self.log, "No old sample information: %s",
                                      self.file_updated)
                new_info = yaml_load(yaml_dump(updated_info_trimmed))
                ignore_blocks = []
                from cobaya.sampler import get_sampler_class_OLD, Minimizer
                if issubclass(get_sampler_class_OLD(new_info) or type, Minimizer):
                    ignore_blocks = [kinds.sampler]
                if not is_equal_info(old_info, new_info, strict=False,
                                     ignore_blocks=ignore_blocks):
                    # HACK!!! NEEDS TO BE FIXED
                    if issubclass(get_sampler_class_OLD(updated_info) or type, Minimizer):
                        # TODO: says work in progress!
                        raise LoggedError(
                            self.log, "Old and new sample information not compatible! "
                                      "At this moment it is not possible to 'force' "
                                      "deletion of and old 'minimize' run. Please delete "
                                      "it by hand. "
                                      "We are working on fixing this very soon!")
                    raise LoggedError(
                        self.log, "Old and new sample information not compatible! "
                                  "Resuming not possible!")
                # Deal with version comparison separately:
                # - If not specified now, take the one used in resumed info
                # - If specified both now and before, check new older than old one
                for k in (kind for kind in kinds if kind in updated_info):
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
                                              "sample that used a newer version: %r.",
                                    new_version, k, c, old_version)
            except IOError:
                # There was no previous chain
                pass
        # We write the new one anyway (maybe updated debug, resuming...)
        for f, info in [(self.file_input, input_info),
                        (self.file_updated, updated_info_trimmed)]:
            if info:
                with open(f, "w", encoding="utf-8") as f_out:
                    try:
                        f_out.write(yaml_dump(info))
                    except OutputError as e:
                        raise LoggedError(self.log, str(e))

    def find_with_regexp(self, regexp):
        """
        Returns all files found which are compatible with this `Output` instance,
        including their path in their name.
        """
        return [
            f2 for f2 in [os.path.join(self.folder, f) for f in os.listdir(self.folder)]
            if f2 == getattr(regexp.match(f2), "group", lambda: None)()]

    def delete_with_regexp(self, regexp):
        """
        Deletes all files compatible with the given regexp.
        """
        file_names = self.find_with_regexp(regexp)
        if file_names:
            self.log.debug("From regexp %r, deleting files %r", regexp.pattern, file_names)
        try:
            [os.remove(f) for f in file_names]
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
            (extension or self.ext))
        return file_name, self.kind

    def collection_regexp(self, name=None, extension=None):
        """
        Returns a regexp for collections compatible with this output settings.

        Use `name` for particular types of collections (default: any number).
        Pass `False` to mean there is nothing between the output prefix and the extension.
        """
        pre = os.path.join(self.folder, self.prefix) + (r"\." if self.prefix else "")
        if name is None:
            name = r"\d+\."
        elif name is False:
            name = ""
        else:
            name = name + r"\."
        extension = extension or self.ext
        return re.compile(self.prefix_regexp_str + name + extension.lower() + "$")

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

        Use `name` for particular types of collections (default: any number).
        Pass `False` to mean there is nothing between the output prefix and the extension.
        """
        return [
            f2 for f2 in [os.path.join(self.folder, f) for f in os.listdir(self.folder)]
            if self.is_collection_file_name(f2, name=name, extension=extension)]

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
                collection._append(collection_i)
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
    MPI wrapper around the Output class.
    """

    def __init__(self, *args, **kwargs):
        if is_main_process():
            Output.__init__(self, *args, **kwargs)
        if more_than_one_process():
            to_broadcast = ("folder", "prefix", "kind", "ext", "resuming")
            values = share_mpi([getattr(self, var) for var in to_broadcast]
                               if is_main_process() else None)
            for name, var in zip(to_broadcast, values):
                setattr(self, name, var)

    def dump_info(self, *args, **kwargs):
        if is_main_process():
            Output.dump_info(self, *args, **kwargs)


def get_output(*args, **kwargs):
    """
    Auxiliary function to retrieve the output driver.
    """
    if kwargs.get("output_prefix"):
        from cobaya.mpi import import_MPI
        return import_MPI(".output", "Output")(*args, **kwargs)
    else:
        return OutputDummy(*args, **kwargs)
