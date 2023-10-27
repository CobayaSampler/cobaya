import os
from io import StringIO
from cobaya.yaml import yaml_load_file
from cobaya.grid_tools.gridconfig import grid_create
from cobaya.grid_tools.gridrun import grid_run
from cobaya.grid_tools.gridmanage import grid_getdist, grid_list, grid_copy
from cobaya.grid_tools.gridtables import grid_tables
from .common import stdout_redirector


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
    assert os.path.exists(os.path.join(f, 'base_a_1_a_2',
                                       'like1', 'base_a_1_a_2_like1.post.cut.1.txt'))

    table_file = os.path.join(tmpdir, 'table')
    grid_tables([f, table_file, '--forpaper'])  # haven't installed latex in general
    assert os.path.exists(table_file + '.tex')

    grid_tables([f, table_file, '--limit', '1', '--forpaper', '--param', 'a_2', '--data',
                 'like2'])
    with open(table_file + '.tex') as r:
        assert '68\\%' in r.read()

    stream = StringIO()
    with stdout_redirector(stream):
        grid_list(f)
    assert "base_like1_like2 (main)" in stream.getvalue()

    stream = StringIO()
    with stdout_redirector(stream):
        grid_copy([f, os.path.join(tmpdir, 'test_grid.zip'), '--dist', '--chains',
                   '--remove_burn_fraction', '0.3', '--paramtag', 'like1_like2'])
    assert "4 dist files" in stream.getvalue()
    assert "1 chain files" in stream.getvalue()
