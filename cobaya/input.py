import warnings
from cobaya import input_processing  # Replace with actual new path

warnings.warn("cobaya.input is deprecated. Use cobaya.input_processing instead.", DeprecationWarning, stacklevel=2)

# Expose everything from the new module
globals().update(vars(input_processing))
