"""
.. module:: output

:Synopsis: Generic output class and output drivers
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import absolute_import, division

# Global
import os
import sys
import six
import traceback
import datetime
from copy import deepcopy
from itertools import chain
import re

# Local
from cobaya.yaml import yaml_dump, yaml_load, yaml_load_file, OutputError
from cobaya.conventions import _input_suffix, _updated_suffix, _separator_files
from cobaya.conventions import _resume, _resume_default, _force, _yaml_extensions
from cobaya.conventions import _likelihood, _params, _sampler
from cobaya.log import LoggedError, HasLogger
from cobaya.input import is_equal_info
from cobaya.mpi import am_single_or_primary_process, get_mpi_comm
from cobaya.collection import Collection
from cobaya.tools import deepcopy_where_possible

# Regular expressions for plain unsigned integers
re_uint = re.compile("[0-9]+")


class Output(HasLogger):
    def __init__(self, output_prefix=None, resume=_resume_default, force_output=False):
        self.name = "output"  # so that the MPI-wrapped class conserves the name
        self.set_logger()
        self.folder = os.sep.join(output_prefix.split(os.sep)[:-1]) or "."
        self.prefix = (lambda x: x if x != "." else "")(output_prefix.split(os.sep)[-1])
        self.force_output = force_output
        if resume and force_output and output_prefix:
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
        if os.path.isfile(self.file_updated):
            self.log.info(
                "Found existing products with the requested ouput prefix: '%s'",
                output_prefix)
            if self.force_output:
                self.log.info("Deleting previous chain ('force' was requested).")
                [os.remove(f) for f in [self.file_input, self.file_updated]]
            elif resume:
                # Only in this case we can be sure that we are actually resuming
                self.resuming = True
                self.log.info("Let's try to resume/load.")
            else:
                # If only input and updated info dumped, overwrite; else fail
                info_files = [
                    os.path.basename(f) for f in [self.file_input, self.file_updated]]
                same_prefix_noinfo = [f for f in os.listdir(self.folder) if
                                      f.startswith(self.prefix) and f not in info_files]
                if not same_prefix_noinfo:
                    [os.remove(f) for f in [self.file_input, self.file_updated]]
                    self.log.info("Overwritten old failed chain files.")
                else:
                    raise LoggedError(
                        self.log, "Delete the previous sample manually, automatically "
                        "('-%s', '--%s', '%s: True')" % (
                            _force[0], _force, _force) +
                        " or request resuming ('-%s', '--%s', '%s: True')" % (
                            _resume[0], _resume, _resume))
        # Output kind and collection extension
        self.kind = "txt"
        self.ext = "txt"

    def updated_output_prefix(self):
        """
        Updated path: drops folder: now it's relative to the chain's location.
        """
        return self.prefix or "."

    def is_resuming(self):
        return self.resuming

    def reload_updated_info(self):
        return yaml_load_file(self.file_updated)

    def dump_info(self, input_info, updated_info, check_compatible=True):
        """
        Saves the info in the chain folder twice:
           - the input info.
           - idem, populated with the modules' defaults.

        If resuming a sample, checks first that old and new infos are consistent.
        """
        # trim known params of each likelihood: for internal use only
        updated_info_trimmed = deepcopy_where_possible(updated_info)
        for lik_info in updated_info_trimmed.get(_likelihood, {}).values():
            if hasattr(lik_info, "pop"):
                lik_info.pop(_params, None)
        if check_compatible:
            try:
                # We will test the old info against the dumped+loaded new info.
                # This is because we can't actually check if python objects do change
                old_info = self.reload_updated_info()
                new_info = yaml_load(yaml_dump(updated_info_trimmed))
                ignore_blocks = []
                if list(new_info.get(_sampler, [None]))[0] == "minimize":
                    ignore_blocks = [_sampler]
                if not is_equal_info(old_info, new_info, strict=False,
                                     ignore_blocks=ignore_blocks):
                    # HACK!!! NEEDS TO BE FIXED
                    if list(updated_info.get(_sampler, [None]))[0] == "minimize":
                        raise LoggedError(
                            self.log, "Old and new sample information not compatible! "
                            "At this moment it is not possible to 'force' deletion of "
                            "and old 'minimize' run. Please delete it by hand. "
                            "We are working on fixing this very soon!")
                    raise LoggedError(
                        self.log, "Old and new sample information not compatible! "
                        "Resuming not possible!")
            except IOError:
                # There was no previous chain
                pass
        # We write the new one anyway (maybe updated debug, resuming...)
        for f, info in [(self.file_input, input_info),
                        (self.file_updated, updated_info_trimmed)]:
            if not info:
                pass
            with open(f, "w") as f_out:
                try:
                    f_out.write(yaml_dump(info))
                except OutputError as e:
                    raise LoggedError(self.log, str(e))

    def prepare_collection(self, name=None, extension=None):
        """
        Generates a file name for the collection, as
        ``[folder]/[prefix].[name].[extension]``.

        Notice that ``name=None`` generates a date, but ``name=""`` removes the ``name``
        field, making it simply ``[folder]/[prefix].[extension]``.
        """
        if name is None:
            name = (datetime.datetime.now().isoformat()
                        .replace("T", "").replace(":", "").replace(".", "").replace("-", "")[
                    :(4 + 2 + 2) + (2 + 2 + 2 + 3)])  # up to ms
        file_name = os.path.join(
            self.folder,
            self.prefix + ("." if self.prefix else "") + (name + "." if name else "") +
            (extension or self.ext))
        return file_name, self.kind

    def is_collection_file_name(self, file_name, extension=None):
        extension = extension or self.ext
        # 1 field only: a number between prefix and extension, ignoring "_" and "."
        fields = list(chain(
            *[_.split("_") for _ in
              file_name[len(self.prefix):-len(extension)].split(".")]))
        fields = [f for f in fields if f]
        return (len(fields) == 1 and
                fields[0] == getattr(re_uint.match(fields[0]), "group", lambda: None)())

    def find_collections(self, extension=None):
        """Returns collection files, including their path."""
        # Anything that looks like a collection
        extension = extension or self.ext
        file_names = [
            os.path.join(self.folder, f) for f in os.listdir(self.folder) if (
                    f.startswith(self.prefix) and
                    f.lower().endswith(extension.lower()) and
                    self.is_collection_file_name(f, extension=extension))]
        return file_names

    def load_collections(self, model, skip=0, thin=1, concatenate=False):
        filenames = self.find_collections()
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
        _func_name = "__name__" if six.PY3 else "func_name"
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
        to_broadcast = ("folder", "prefix", "kind", "ext", "resuming")
        if am_single_or_primary_process():
            Output.__init__(self, *args, **kwargs)
        else:
            for var in to_broadcast:
                setattr(self, var, None)
        for var in to_broadcast:
            setattr(self, var, get_mpi_comm().bcast(getattr(self, var), root=0))

    def dump_info(self, *args, **kwargs):
        if am_single_or_primary_process():
            Output.dump_info(self, *args, **kwargs)


def get_Output(*args, **kwargs):
    """
    Auxiliary function to retrieve the output driver.
    """
    if kwargs.get("output_prefix"):
        from cobaya.mpi import import_MPI
        Output = import_MPI(".output", "Output")
        return Output(*args, **kwargs)
    else:
        return OutputDummy(*args, **kwargs)
