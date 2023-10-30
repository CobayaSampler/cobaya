from io import StringIO
import os
from cobaya.doc import doc_script
from cobaya.bib import bib_script
from cobaya.grid_tools import gridmanage, grid_create, grid_run

from cobaya.yaml import yaml_load_file
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


def test_cosmo_grid(tmpdir):
    test_name = 'base_w_planck_lowl_NPIPE_TTTEEE_lensing'
    f = os.path.join(tmpdir, 'grid')
    grid_create([f, os.path.join(os.path.dirname(__file__), 'test_cosmo_grid.yaml')])

    info = yaml_load_file(
        os.path.join(f, 'input_files', 'base_mnu_planck_lowl_NPIPE_TT.yaml'))
    assert info['theory']['camb']['extra_args']['num_massive_neutrinos'] == 3

    with stdout_check(test_name):
        grid_run([f, '--dryrun', '--job-template',
                  'cobaya/grid_tools/script_templates/job_script_UGE'])

    with stdout_check(test_name):
        gridmanage.grid_list(f)

    with stdout_check(test_name):
        gridmanage.grid_copy([f, os.path.join(tmpdir, 'grid.zip')])

    with stdout_check("0 MB"):
        gridmanage.grid_cleanup([f])

    with stdout_check("Chains do not exist yet"):
        gridmanage.grid_getdist([f])

    with stdout_check("Chains do not", match=False):
        gridmanage.grid_getdist([f, '--exist'])
