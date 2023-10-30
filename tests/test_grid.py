import os
from cobaya.yaml import yaml_load_file
from cobaya.grid_tools import grid_create, grid_run, grid_converge, grid_tables, \
    grid_param_compare, grid_getdist, grid_list, grid_copy, grid_extract, grid_cleanup
from .common import stdout_check


def test_grid(tmpdir):

    f = os.path.join(tmpdir, 'grid')
    grid_create([f, os.path.join(os.path.dirname(__file__), 'simple_grid.py')])
    assert os.path.exists(os.path.join(f, 'base', 'like1_like2'))
    assert os.path.exists(os.path.join(f, 'input_files', 'base_a_2_like1_like2.yaml'))
    assert not os.path.exists(os.path.join(f, 'input_files', 'base_a_1_like1_like2.yaml'))

    info = yaml_load_file(
        os.path.join(f, 'input_files', 'base_a_1_a_2_like1_like2.yaml'))
    assert info['sampler']['mcmc']['max_samples'] == 100

    grid_run([f, '--noqueue'])
    grid_run([f, '--noqueue', '--importance'])
    grid_getdist([f, '--burn_remove', '0.3'])
    assert os.path.exists(os.path.join(f, 'base_a_1_a_2',
                                       'like1', 'dist', 'base_a_1_a_2_like1.margestats'))
    assert os.path.exists(os.path.join(f, 'base_a_1_a_2', 'like1_like2',
                                       'base_a_1_a_2_like1_like2.post.cut.1.txt'))

    grid_run([f, '--noqueue', '--minimize', '--name', 'base_a_2_like1'])
    assert os.path.exists(os.path.join(f, 'base_a_2', 'like1', 'base_a_2_like1.minimum'))

    table_file = os.path.join(tmpdir, 'table')
    grid_tables([f, table_file, '--forpaper'])  # haven't installed latex in general
    assert os.path.exists(table_file + '.tex')

    grid_tables([f, table_file, '--limit', '1', '--forpaper',
                 '--param', 'a_2', '--data', 'like2'])
    with open(table_file + '.tex') as r:
        assert '68\\%' in r.read()

    with stdout_check("base_like1_like2 (main)"):
        grid_list(f)

    with stdout_check("10 dist files", "1 chain file"):
        grid_copy([f, os.path.join(tmpdir, 'test_grid.zip'), '--dist', '--chains',
                   '--remove_burn_fraction', '0.3', '--datatag', 'like1_like2'])

    with stdout_check("base_a_2_like1_like2 None"):
        grid_converge([f])

    with stdout_check("base_like1_like2"):
        grid_converge([f, '--checkpoint'])

    grid_param_compare([f, '--params', 'a_1', 'a_2', '--latex_filename', table_file])
    with open(table_file + '.tex') as r:
        assert 'a_2 &' in r.read()

    grid_extract([f, tmpdir, '.margestats', '--datatag', 'like1_like2'])
    assert os.path.exists(os.path.join(tmpdir, 'base_a_1_a_2_like1_like2.margestats'))

    with stdout_check("base_a_1_a_2_like1_like2"):
        grid_cleanup([f, '--confirm', '--data', 'like2'])
