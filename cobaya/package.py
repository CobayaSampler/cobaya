"""
Static module containing packaging information for use in CI/CD or at docs building stage,
when Cobaya has not yet been installed as a package, to avoid missing dependencies, e.g.

    from cobaya.package import __version__

instead of

    from cobaya import __version__
"""

__author__ = "Jesus Torrado and Antony Lewis"
__version__ = "3.3.2"
__obsolete__ = False
__year__ = "2023"
__url__ = "https://cobaya.readthedocs.io"
