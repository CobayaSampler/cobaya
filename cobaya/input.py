import warnings
from cobaya import cobaya_input  # Replace with actual new path

warnings.warn("cobaya.input is deprecated. Use cobaya.cobaya_input instead.", DeprecationWarning, stacklevel=2)

# Expose everything from the new module
globals().update(vars(cobaya_input))
