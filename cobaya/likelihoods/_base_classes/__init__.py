from ..base_classes import InstallableLikelihood, DataSetLikelihood, CMBlikes

warn_msg = ("*DEPRECATION*: Likelihood class '_base_classes._{name}' has been renamed to "
            "'base_classes.{name}' (no leading underscores!). Please use the new name,"
            " since the old one will be deprecated in the near future")


class _InstallableLikelihood(InstallableLikelihood):

    def __init__(self, *args, **kwargs):
        # MARKED FOR DEPRECATION IN v3.0
        raise ValueError(warn_msg.format(name="InstallableLikelihood"))
        # END OF DEPRECATION BLOCK


class _DataSetLikelihood(DataSetLikelihood):

    def __init__(self, *args, **kwargs):
        # MARKED FOR DEPRECATION IN v3.0
        raise ValueError(warn_msg.format(name="DataSetLikelihood"))
        # END OF DEPRECATION BLOCK


class _CMBlikes(CMBlikes):

    def __init__(self, *args, **kwargs):
        # MARKED FOR DEPRECATION IN v3.0
        raise ValueError(warn_msg.format(name="CMBlikes"))
        # END OF DEPRECATION BLOCK
