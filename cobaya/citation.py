"""
.. module:: citation

:Synopsis: Old name for module ``bib``. Will be deprecated.
:Author: Jesus Torrado

"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

# Local
from cobaya.tools import warn_deprecation, create_banner
from cobaya.bib import bib_script


# Command-line script
def citation_script():
    warn_deprecation()
    print(create_banner(
        "\nThis command will be deprecated soon. Use `cobaya-bib` instead.\n"))
    bib_script()
