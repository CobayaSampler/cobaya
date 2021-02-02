import sys
from ..base_classes import InstallableLikelihood, DataSetLikelihood, CMBlikes


warn_msg = "*WARNING*: Likelihood class '_base_classes._{name}' has been renamed to " \
           "'base_classes.{name}' (no leading underscores!). Please use the new name," \
           " since the old one will be deprecated in the near future"


class _InstallableLikelihood(InstallableLikelihood):

    def __init__(self, *args, **kwargs):
        # MARKED FOR DEPRECATION IN v3.0
        print(warn_msg.format(name="InstallableLikelihood"), file=sys.stderr)
        # BEHAVIOUR TO BE REPLACED BY ERROR:
        super().__init__(*args, **kwargs)
        # END OF DEPRECATION BLOCK


class _DataSetLikelihood(DataSetLikelihood):

    def __init__(self, *args, **kwargs):
        # MARKED FOR DEPRECATION IN v3.0
        print(warn_msg.format(name="DataSetLikelihood"), file=sys.stderr)
        # BEHAVIOUR TO BE REPLACED BY ERROR:
        super().__init__(*args, **kwargs)
        # END OF DEPRECATION BLOCK


class _CMBlikes(CMBlikes):

    def __init__(self, *args, **kwargs):
        # MARKED FOR DEPRECATION IN v3.0
        print(warn_msg.format(name="CMBlikes"), file=sys.stderr)
        # BEHAVIOUR TO BE REPLACED BY ERROR:
        super().__init__(*args, **kwargs)
        # END OF DEPRECATION BLOCK
