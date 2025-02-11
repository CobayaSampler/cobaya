import warnings
from cobaya import yaml_helpers  # Replace with actual new path

warnings.warn("cobaya.yaml is deprecated. Use cobaya.yaml_helpers instead.", DeprecationWarning, stacklevel=2)

# Expose everything from the new module
globals().update(vars(yaml_helpers))
