# Tries to evaluate the likelihood at LCDM's best fit of Planck 2015, with CLASS

from cosmo_common import body_of_test

def test_classy_planck_t(modules):
    body_of_test(modules, "t", "classy")

def test_classy_planck_p(modules):
    body_of_test(modules, "p", "classy")
