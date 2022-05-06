from io import StringIO
import os
from cobaya.doc import doc_script
from cobaya.bib import bib_script
from .common import stdout_redirector


def test_doc():
    stream = StringIO()
    with stdout_redirector(stream):
        doc_script([])
    assert 'planck_2018_lensing' in stream.getvalue()
    stream = StringIO()
    with stdout_redirector(stream):
        doc_script(['des_y1.shear'])
    assert 'DES_DzS1' in stream.getvalue()

    stream = StringIO()
    with stdout_redirector(stream):
        doc_script(['planck_2018_highl_CamSpec.TT'])
    assert 'params: !defaults [params_calib_CamSpec, ' \
           'params_TT_CamSpec, params_TT_CamSpec_fixedcalpol]' in stream.getvalue()
    stream = StringIO()
    with stdout_redirector(stream):
        doc_script(['planck_2018_highl_CamSpec.TT', '--expand'])
    assert 'lambda aksz, asz143' in stream.getvalue()


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
    stream = StringIO()
    with stdout_redirector(stream):
        bib_script(['des_y1.shear', 'camb', 'mcmc'])
    assert 'Neal:2005' in stream.getvalue()

    f = os.path.join(tmpdir, 'input.yaml')
    with open(f, 'w', encoding='utf-8') as output:
        output.write(yaml)

    stream = StringIO()
    with stdout_redirector(stream):
        bib_script([f])
    assert 'Torrado:2020' in stream.getvalue()
