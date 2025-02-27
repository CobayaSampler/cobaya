import warnings
from cobaya import typing_conventions  # Replace with actual new path

warnings.warn("cobaya.typing is deprecated. Use cobaya.typing_conventions instead.", DeprecationWarning, stacklevel=2)

# Expose everything from the new module
globals().update(vars(typing_conventions))
