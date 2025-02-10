import warnings
from cobaya import cobaya_typing  # Replace with actual new path

warnings.warn("cobaya.typing is deprecated. Use cobaya.cobaya_typing instead.", DeprecationWarning, stacklevel=2)

# Expose everything from the new module
globals().update(vars(cobaya_typing))
