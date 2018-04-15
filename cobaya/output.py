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
import traceback
import datetime

# Local
from cobaya.yaml import yaml_dump
from cobaya.conventions import _input_suffix, _full_suffix, separator, _yaml_extension
from cobaya.conventions import _output_prefix, _force_reproducible
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__.split(".")[-1])


class Output(object):
    def __init__(self, info):
        output_prefix = str(info[_output_prefix])
        self.force_reproducible = info.get(_force_reproducible, True)
        self.folder = os.sep.join(output_prefix.split(os.sep)[:-1]) or "."
        self.prefix = (lambda x: x if x != "." else "")(output_prefix.split(os.sep)[-1])
        if not os.path.exists(self.folder):
            log.debug("Creating output folder '%s'", self.folder)
            try:
                os.makedirs(self.folder)
            except OSError:
                log.error("".join(["-"]*20 + ["\n\n"] +
                                  list(traceback.format_exception(*sys.exc_info())) +
                                  ["\n"] + ["-"]*37))
                log.error("Could not create folder '%s'. "
                          "See traceback on top of this message.", self.folder)
                raise HandledException
        log.info("Products to be written into folder '%s', with prefix '%s'.",
                 self.folder, self.prefix)
        # Prepare file names, and check if chain exists
        self.file_prefix = self.prefix + (separator if self.prefix else "")
        self.file_input = os.path.join(
            self.folder, self.file_prefix+_input_suffix+_yaml_extension)
        self.file_full = os.path.join(
            self.folder, self.file_prefix+_full_suffix+_yaml_extension)
        if os.path.isfile(self.file_full):
            log.error("Chain continuation not implemented, sorry. "
                      "Delete the previous chain and try again, "
                      "or choose a different output prefix.")
            raise HandledException
        # Save the updated output_prefix: now relative to output folder
        info[_output_prefix] = self.prefix or "."
        # Output kind and collection extension
        self.kind = "txt"
        self.ext = "txt"

    def dump_info(self, input_info, full_info):
        """
        Saves the info in the chain folder twice:
           - the input info.
           - idem, populated with the modules' defaults.
        """
        for f, info in [(self.file_input, input_info),
                        (self.file_full, full_info)]:
            with open(f, "w") as f_out:
                f_out.write(
                    yaml_dump(info, default_flow_style=False, trim_params_info=True,
                              force_reproducible=self.force_reproducible))

    def prepare_collection(self, name=None, extension=None):
        if not name:
            name = (datetime.datetime.now().isoformat()
                    .replace("T","_").replace(":","-").replace(".","-"))
        file_name = os.path.join(
            self.folder,
            self.prefix+("_" if self.prefix else "")+name+"."+(extension or self.ext))
        return file_name, self.kind


class Output_MPI(Output):
    """
    MPI wrapper around the Output class.
    """
    def __init__(self, info):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        to_broadcast = ("folder", "prefix", "kind", "ext")
        if rank == 0:
            Output.__init__(self, info)
        else:
            for var in to_broadcast:
                setattr(self, var, None)
        for var in to_broadcast:
            setattr(self, var, comm.bcast(getattr(self, var), root=0))

    def dump_info(self, *args, **kwargs):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            Output.dump_info(self, *args, **kwargs)


class Output_dummy(Output):
    """
    Dummy output class. Does nothing. Evaluates to 'False' as a class.
    """
    def __init__(self, info):
        log.info("No output requested. Doing nothing (or returning in scripted call).")
        # override all methods that actually produce output
        exclude = ["nullfunc"]
        if sys.version_info < (3,):
            _func_name = "func_name"
        else:
            _func_name = "__name__"
        for attrname,attr in list(Output.__dict__.items()):
            func_name = getattr(attr, _func_name, None)
            if func_name and func_name not in exclude and '__' not in func_name:
                setattr(self, attrname, self.nullfunc)

    def nullfunc(self, *args, **kwargs):
        pass

    def __nonzero__(self):
        return False

    def __bool__(self):
        return False
