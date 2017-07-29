"""
.. module:: install

:Synopsis: Tools and script to install the modules requested in the given input.
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division

# Global
import os

# Local
from cobaya.tools import get_folder
from cobaya.input import get_modules



# los scripts de instalacion tienene que tener un test para ver si ya esta instalado
# (no necesariamente demasiado fiable) y un opci'on ("force") para ignorar el test.


def get_citation(module, kind):
    folder = get_folder(module, kind, absolute=True)
    filename = os.path.join(folder, module+".bibtex")
    try:
        with open(filename, "r") as f:
            lines = "".join(f.readlines())
    except IOError:
        if not os.path.isdir(folder):
            lines = "[Module '%s.%s' not known.]"%(kind, module)
        else:
            lines = "[no citation information found]"
    header = "\n"+kind.title()+" : "+module
    header += "\n"+"="*(len(header)-1)+"\n\n"
    return header+lines+"\n"

def citation(*infos):
    header = "This framework:"
    header += "\n"+"="*len(header)+"\n\n"
    print "\n"+header+"[No citation information for now]\n"
    for kind, modules in get_modules(*infos).iteritems():
        for module in modules:
            print get_citation(module, kind)

# Command-line script
def citation_script():
    from cobaya.mpi import import_MPI
    load_input = import_MPI(".input", "load_input")
    import sys
    infos = [load_input(filename) for filename in sys.argv[1:]]
    citation(*infos)
