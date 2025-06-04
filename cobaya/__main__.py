"""
Allows calling `cobaya-[command]` as `python -m cobaya [command]`.

`run` is optional: one can pass directly an input file as
`python -m cobaya input.yaml`.
"""

import sys
from importlib import import_module, metadata


def run_command():
    commands = {}
    prefix = "cobaya-"
    console_scripts = (
        metadata.entry_points().select(group="console_scripts")
        if sys.version_info >= (3, 10)
        else metadata.entry_points()["console_scripts"]
    )
    for script in console_scripts:
        if script.name.startswith(prefix):
            commands[script.name] = script.value
    commands_trimmed = [c[len(prefix) :] for c in commands]
    help_msg = (
        "Add a one of the following commands and its arguments "
        f"(`<command> -h` for help): {commands_trimmed}"
    )
    try:
        command_or_input = sys.argv[1].lower()
    except IndexError:  # no command
        print(help_msg)
    else:
        command = commands.get("cobaya-" + command_or_input)
        if command is not None:
            module, func = command.split(":")
            sys.argv.pop(1)
            assert func is not None
            getattr(import_module(module), func)()
        else:
            if command_or_input in ["-h", "--help"]:
                print(help_msg)
            else:
                # no command --> assume run with input file as 1st arg (don't pop!)
                getattr(import_module("cobaya.run"), "run_script")()


if __name__ == "__main__":
    run_command()
