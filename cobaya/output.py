"""
.. module:: output

:Synopsis: Generic output class and output drivers
:Author: Jesus Torrado

"""

# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os
import sys
import six
import traceback
import datetime
from copy import deepcopy

# Local
from cobaya.yaml import yaml_dump, yaml_load, yaml_load_file, OutputError
from cobaya.conventions import _input_suffix, _full_suffix, _separator, _yaml_extensions
from cobaya.conventions import _resume, _resume_default, _force
from cobaya.conventions import _likelihood, _params
from cobaya.log import HandledException
from cobaya.input import is_equal_info
from cobaya.mpi import am_single_or_primary_process, get_mpi_comm

# Logger
import logging


class Output(object):
    def __init__(self, output_prefix=None, resume=_resume_default, force_output=False):
        self.log = logging.getLogger("output")
        self.folder = os.sep.join(output_prefix.split(os.sep)[:-1]) or "."
        self.prefix = (lambda x: x if x != "." else "")(output_prefix.split(os.sep)[-1])
        self.force_output = force_output
        if resume and force_output and output_prefix:
            # No resume and force at the same time (if output)
            self.log.error(
                "Make 'resume: True' or 'force: True', not both at the same time: "
                "can't simultaneously overwrite a chain and resume from it.")
            raise HandledException
        if not os.path.exists(self.folder):
            self.log.debug("Creating output folder '%s'", self.folder)
            try:
                os.makedirs(self.folder)
            except OSError:
                self.log.error("".join(["-"] * 20 + ["\n\n"] +
                                       list(traceback.format_exception(*sys.exc_info())) +
                                       ["\n"] + ["-"] * 37))
                self.log.error("Could not create folder '%s'. "
                               "See traceback on top of this message.", self.folder)
                raise HandledException
        self.log.info("Products to be written into folder '%s', with prefix '%s'",
                      self.folder, self.prefix)
        # Prepare file names, and check if chain exists
        info_file_prefix = os.path.join(
            self.folder, self.prefix + (_separator if self.prefix else ""))
        self.file_input = info_file_prefix + _input_suffix + _yaml_extensions[0]
        self.file_full = info_file_prefix + _full_suffix + _yaml_extensions[0]
        self.resuming = False
        if os.path.isfile(self.file_full):
            self.log.info(
                "Found an existing sample with the requested ouput prefix: '%s'",
                output_prefix)
            if self.force_output:
                self.log.info("Deleting previous chain ('force' was requested).")
                [os.remove(f) for f in [self.file_input, self.file_full]]
            elif resume:
                # Only in this case we can be sure that we are actually resuming
                self.resuming = True
                self.log.info("Let's try to resume sampling.")
            else:
                # If only input and full info dumped, overwrite; else fail
                info_files = [
                    os.path.basename(f) for f in [self.file_input, self.file_full]]
                same_prefix_noinfo = [f for f in os.listdir(self.folder) if
                                      f.startswith(self.prefix) and f not in info_files]
                if not same_prefix_noinfo:
                    [os.remove(f) for f in [self.file_input, self.file_full]]
                    self.log.info("Overwritten old failed chain files.")
                else:
                    self.log.error("Delete the previous sample manually, automatically "
                                   "('-%s', '--%s', '%s: True')" % (
                                       _force[0], _force, _force) +
                                   " or request resuming ('-%s', '--%s', '%s: True')" % (
                                       _resume[0], _resume, _resume))
                    raise HandledException
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

    def dump_info(self, input_info, full_info):
        """
        Saves the info in the chain folder twice:
           - the input info.
           - idem, populated with the modules' defaults.

        If resuming a sample, checks first that old and new infos are consistent.
        """
        # trim known params of each likelihood: for internal use only
        full_info_trimmed = deepcopy(full_info)
        for lik_info in full_info_trimmed.get(_likelihood, {}).values():
            if hasattr(lik_info, "pop"):
                lik_info.pop(_params, None)
        try:
            # We will test the old info agains the dumped+loaded new info.
            # This is because we can't actually check if python objects are the same as before.
            old_info = yaml_load_file(self.file_full)
            new_info = yaml_load(yaml_dump(full_info_trimmed))
            if not is_equal_info(old_info, new_info, strict=False):
                self.log.error("Old and new sample information not compatible! "
                               "Resuming not possible!")
                raise HandledException
        except IOError:
            # There was no previous chain
            pass
        # We write the new one anyway (maybe updated debug, resuming...)
        for f, info in [(self.file_input, input_info),
                        (self.file_full, full_info_trimmed)]:
            with open(f, "w") as f_out:
                try:
                    f_out.write(yaml_dump(info))
                except OutputError as e:
                    self.log.error(e.message)
                    raise HandledException

    def prepare_collection(self, name=None, extension=None):
        if not name:
            name = (datetime.datetime.now().isoformat()
                    .replace("T", "_").replace(":", "-").replace(".", "-"))
        file_name = os.path.join(
            self.folder,
            self.prefix + ("_" if self.prefix else "") + name + "." + (extension or self.ext))
        return file_name, self.kind


class OutputDummy(Output):
    """
    Dummy output class. Does nothing. Evaluates to 'False' as a class.
    """

    def __init__(self, *args, **kwargs):
        self.log = logging.getLogger("output")
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
