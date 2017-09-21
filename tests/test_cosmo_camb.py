# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CAMB

from cosmo_common import body_of_test

def test_camb_planck_t(modules):
    body_of_test(modules, "t", "camb")
    
def test_camb_planck_p(modules):
    body_of_test(modules, "p", "camb")
