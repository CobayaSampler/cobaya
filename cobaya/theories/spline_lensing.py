import os
from cobaya.theories.camb import CAMB
from cobaya.theory import Theory
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np


class SplineLensing(Theory):
    nodes = np.log(np.array([7, 44, 125, 600, 1600, 3100]))

    def spectrum(self, amps, lmax):
        sp = spline(self.nodes, amps)
        logl = np.log(np.arange(2, lmax + 1))
        return np.concatenate(([0, 0], np.exp(sp(logl))))

    def get_Cpp(self, lmax):
        amps = self.provider.get_param('p%s' % i for i in range(len(self.nodes)))
        return {'pp': self.spectrum(amps, lmax)}

    def get_ISW(self, lmax):
        return np.concatenate(
            ([0, 0], self.provider.get_param('Aisw') / np.arange(2, lmax + 1) ** 2))

    @classmethod
    def get_class_options(cls, input_options={}):
        opts = super().get_class_options(input_options)
        opts['params'] = {'p%s' % i: {'prior': {'min': -22, 'max': -14}, 'proposal': 0.1}
                          for i in range(len(cls.nodes))}
        opts['params']['Aisw'] = {'prior': {'min': 0, 'max': 3}, 'proposal': 0.02,
                                  'ref': 0.1}
        return opts


class PlanckSplineLensing(SplineLensing):
    nodes = np.log(np.array([7, 44, 125, 600, 1600]))


class CambSpline(CAMB):

    def must_provide(self, **requirements):
        req = super().must_provide(**requirements)
        req['Cpp'] = None
        req['ISW'] = None
        self.collectors['CAMBdata'] = None
        return req

    def calculate(self, state, want_derived=True, **params_values_dict):
        super().calculate(state, want_derived, **params_values_dict)
        CAMBdata = state['CAMBdata']
        clpp = self.provider.get_Cpp(lmax=CAMBdata.Params.max_l)

        cls_array = CAMBdata.get_lensed_cls_with_spectrum(clpp['pp'], raw_cl=False)
        ISW = self.provider.get_ISW(lmax=cls_array.shape[0] - 1)
        cls_array[:, 0] = cls_array[:, 0] + cls_array[30, 0] * 30 ** 2 * ISW
        state['Cl']['total'] = cls_array
        cp = np.empty((cls_array.shape[0], 1), dtype=np.float64)
        cp[:, 0] = clpp["pp"][0:cls_array.shape[0]]
        state['Cl']['lens_potential'] = cp
