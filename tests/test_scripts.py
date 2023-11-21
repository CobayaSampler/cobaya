import os
from cobaya.doc import doc_script
from cobaya.bib import bib_script
from .common import stdout_check


def test_doc():
    with stdout_check('planck_2018_lensing'):
        doc_script([])

    with stdout_check('DES_DzS1'):
        doc_script(['des_y1.shear'])

    with stdout_check(' [../base_classes/planck_calib, params_TT_CamSpec]'):
        doc_script(['planck_2018_highl_CamSpec.TT'])

    with stdout_check('lambda aksz, asz143'):
        doc_script(['planck_2018_highl_CamSpec.TT', '--expand'])


yaml = """
likelihood:
 des_y1.clustering:
 bao.generic:
 planck_2018_lensing.clik:
theory:
 camb:
sampler:
 minimize:
"""


def test_bib(tmpdir):
    with stdout_check('Neal:2005'):
        bib_script(['des_y1.shear', 'camb', 'mcmc'])

    f = os.path.join(tmpdir, 'input.yaml')
    with open(f, 'w', encoding='utf-8') as output:
        output.write(yaml)

    with stdout_check('Torrado:2020'):
        bib_script([f])
