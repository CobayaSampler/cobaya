import warnings
from cobaya import cobaya_yaml  # Replace with actual new path

warnings.warn("cobaya.yaml is deprecated. Use cobaya.cobaya_yaml instead.", DeprecationWarning, stacklevel=2)

# Expose everything from the new module
globals().update(vars(cobaya_yaml))
