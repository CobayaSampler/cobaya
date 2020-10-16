"""
Allows calling `cobaya-[command]` as `python -m cobaya [command]`.

`run` is optional: one can pass directly an input file as
`python -m cobaya input.yaml`.
"""

import sys
from importlib import import_module

commands = {"install": ["install", "install_script"],
            "doc": ["doc", "doc_script"],
            "bib": ["bib", "bib_script"],
            "run": ["run", "run_script"],
            "cosmo-generator": ["cosmo_input", "gui_script"],
            "create-image": ["containers", "create_image_script"],
            "prepare-data": ["containers", "prepare_data_script"],
            "grid-create": ["grid_tools", "make_grid_script"],
            "grid-run": ["grid_tools.runbatch", "run"],
            "run-job": ["grid_tools.runMPI", "run_single"],
            }


help_msg = ("Add a one of the following commands and its arguments "
            "(`<command> -h` for help): %r" % list(commands))

if __name__ == "__main__":

    try:
        command_or_input = sys.argv[1].lower()
    except IndexError:  # no command
        print(help_msg)
        exit()

    module, func = commands.get(command_or_input, (None, None))

    if module is not None:
        sys.argv.pop(1)
        getattr(import_module("cobaya." + module), func)()
    else:
        if command_or_input in ["-h", "--help"]:
            print(help_msg)
            exit()
        else:
            # no command --> assume run with input file as 1st arg (don't pop!)
            module, func = commands["run"]
            getattr(import_module("cobaya." + module), func)(
                help_commands=str(list(commands)))
