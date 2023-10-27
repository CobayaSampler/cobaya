from io import StringIO
import os
from cobaya.doc import doc_script
from cobaya.bib import bib_script
from cobaya.grid_tools.gridconfig import grid_create
from cobaya.grid_tools.gridrun import grid_run
from cobaya.grid_tools import gridmanage

from cobaya.yaml import yaml_load_file
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
    assert ' [../base_classes/planck_calib, params_TT_CamSpec]' in stream.getvalue()
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


def test_cosmo_grid(tmpdir):
    test_name = 'base_w_planck_lowl_NPIPE_TTTEEE_lensing'
    f = os.path.join(tmpdir, 'grid')
    grid_create([f, os.path.join(os.path.dirname(__file__), 'test_cosmo_grid.yaml')])

    info = yaml_load_file(
        os.path.join(f, 'input_files', 'base_mnu_planck_lowl_NPIPE_TT.yaml'))
    assert info['theory']['camb']['extra_args']['num_massive_neutrinos'] == 3
    stream = StringIO()
    with stdout_redirector(stream):
        grid_run([f, '--dryrun', '--job-template',
                  'cobaya/grid_tools/script_templates/job_script_UGE'])
    assert test_name in stream.getvalue()

    stream = StringIO()
    with stdout_redirector(stream):
        gridmanage.grid_list(f)
    assert test_name in stream.getvalue()

    stream = StringIO()
    with stdout_redirector(stream):
        gridmanage.grid_copy([f, os.path.join(tmpdir, 'grid.zip')])
    assert test_name in stream.getvalue()

    stream = StringIO()
    with stdout_redirector(stream):
        gridmanage.grid_cleanup([f])
    assert "0 MB" in stream.getvalue()

    stream = StringIO()
    with stdout_redirector(stream):
        gridmanage.grid_getdist([f])
    assert "Chains do not exist yet" in stream.getvalue()

    stream = StringIO()
    with stdout_redirector(stream):
        gridmanage.grid_getdist([f, '--exist'])
    assert "Chains do not" not in stream.getvalue()
