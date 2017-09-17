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
from copy import deepcopy
import datetime
        
# Local
from cobaya.yaml_custom import yaml_dump
from cobaya.conventions import _likelihood, _theory, _prior, _params
from cobaya.conventions import _sampler, _input_suffix, _full_suffix
from cobaya.conventions import separator
from cobaya.log import HandledException

# Logger
import logging
log = logging.getLogger(__name__)

class Output():
    def __init__(self, info):
        output_prefix = str(info["output_prefix"])
        if os.sep in output_prefix:
            self.folder = os.sep.join(output_prefix.split(os.sep)[:-1])
            self.prefix = output_prefix.split(os.sep)[-1]
            log.debug("Creating output folder '%s'", self.folder)
            # Notice that the folder cannot exist already.
            # If we are continuing a sample, that is dealt with by the invocation script.
            try:
                os.makedirs(self.folder)
            except OSError:
                log.error("Chain continuation not implemented. "
                          "If testing, delete the chain folder '%s' before invoking again.",
                          self.folder)
                raise HandledException
                # The exception that should be raised when this is implemented.
                raise OSError("Cannot create folder '%s'. Check your 'output_prefix'.", self.folder)
        else:
            self.folder = "."
            # safeguard against calling from chain folder
            if output_prefix == ".":
                output_prefix = ""
            self.prefix = output_prefix
        log.info("The output folder is '%s', and the output prefix is '%s'", 
                 self.folder, self.prefix)
        # Save the updated name and output_prefix: now relative to output folder
        self.info_input = deepcopy(info)
        self.info_input["output_prefix"] = self.prefix if self.prefix else "."
        # Output kind and collection extension
        self.kind = "txt"
        self.ext  = "txt"

    def set_full_info(self, full_info):
        self._full_info = full_info

    def dump_info(self, likelihood=None, theory=None, sampler=None, params=None):
        """
        Saves the info in the chain folder twice:
           - the input info.
           - idem, populated with the modules' defaults.
        """
        file_prefix = self.prefix
        if self.prefix:
            file_prefix += separator
        file_input = os.path.join(self.folder, file_prefix+_input_suffix+".yaml")
        file_full = os.path.join(self.folder, file_prefix+_full_suffix+".yaml")
        for f, info in [(file_input, self.info_input), (file_full, self._full_info)]:
            if os.path.isfile(f):
                log.error("Chain continuation not implemented. "
                          "If testing, delete the relevant chain folder.")
                raise HandledException
                # This is the error we should be raising when chain continuation is implem.
                raise OSError("Dumping the info: The file '%s' exists!", f)
            with open(f, "w") as f_out:
                f_out.write(yaml_dump(info, default_flow_style=False))

    def prepare_collection(self, name=None):
        if not name:
            name = (datetime.datetime.now().isoformat()
                    .replace("T","_").replace(":","-").replace(".","-"))
        file_name = os.path.join(
            self.folder, self.prefix+("_" if self.prefix else "")+name+"."+self.ext)
        return file_name, self.kind


class Output_MPI(Output):
    """
    MPI wrapper around the Output class.
    """
    def __init__(self, info):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        to_broadcast = ("folder", "prefix", "kind", "ext", "info_input")
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
        self.info_input = deepcopy(info)
        # override all methods
        exclude = ["__nonzero__", "nullfunc", "update_info", "updated_info"]
        for attrname,attr in Output.__dict__.items():
            func_name = getattr(attr, "func_name", None)
            if func_name and not func_name in exclude:
                setattr(self, attrname, self.nullfunc)

    def nullfunc(self, *args, **kwargs):
        pass

    def __nonzero__(self):
        return False
