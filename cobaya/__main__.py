"""
Allows calling `cobaya-[command]` as `python -m cobaya [command]`.

`run` is optional: one can pass directly an input file as
`python -m cobaya input.yaml`.
"""
import sys
from importlib import import_module, metadata

if __name__ == "__main__":
    commands = {}
    for script in metadata.entry_points().select(group='console_scripts'):
        if script.name.startswith('cobaya-'):
            commands[script.name] = script.value
    help_msg = ("Add a one of the following commands and its arguments "
                "(`<command> -h` for help): %r" % list(commands))

    try:
        command_or_input = sys.argv[1].lower()
    except IndexError:  # no command
        print(help_msg)
    else:
        if command := commands.get("cobaya-" + command_or_input):
            module, func = command.split(":")
            sys.argv.pop(1)
            assert func is not None
            getattr(import_module(module), func)()
        else:
            if command_or_input in ["-h", "--help"]:
                help_msg = ("Add a one of the following commands and its arguments "
                            "(`<command> -h` for help): %r" % list(commands))
                print(help_msg)
            else:
                # no command --> assume run with input file as 1st arg (don't pop!)
                getattr(import_module("cobaya.run"), "run_script")()
