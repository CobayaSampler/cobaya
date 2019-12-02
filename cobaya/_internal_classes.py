# a module with all the internal component classes dynamically found and put into
# fake modules, e.g. for use by sphinx docs

# e,g. can reference cobaya._internal_classes.likelihood for module of all
# likelihood classes


from cobaya.conventions import kinds
from cobaya.tools import get_available_internal_classes
import types

for k, classes in [(k, get_available_internal_classes(k, True)) for k in kinds]:

    module = types.ModuleType(k)
    globals()[k] = module

    for cls in classes:
        cls.__module__ = module.__name__
        setattr(module, cls.__name__, cls)
        del cls
