# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
from itertools import chain
import re
import sys

subfolders = {"likelihood": "likelihoods", "sampler": "samplers", "theory": "theories"}


def find_version():
    init_file = open(path.join(path.dirname(__file__), 'cobaya', '__init__.py')).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_long_description():
    """Get the long description from the README file."""
    with open(path.join(path.abspath(path.dirname(__file__)), 'README.rst'),
              encoding='utf-8') as f:
        lines = f.readlines()
        i = -1
        while '=====' not in lines[i]:
            i -= 1
        return "".join(lines[:i])


def extract_docs_requirements():
    """Get requirements for building the documentation."""
    path_docs_requirements = path.join(path.abspath(path.dirname(__file__)),
                                       "docs/requirements.txt")
    with open(path_docs_requirements, "r") as f:
        lines = f.readlines()
    i_first = next(i for i, line in enumerate(lines)
                   if line.strip().startswith("-e")) + 1
    reqs = [line.strip() for line in lines[i_first:]]
    return ["sphinx"] + reqs


install_requires = ['numpy>=1.17.0', 'scipy>=1.5', 'pandas>=1.0.1,!=1.4.0',
                    'PyYAML>=5.1', 'requests>=2.18', 'py-bobyqa>=1.2',
                    'GetDist>=1.3.1', 'fuzzywuzzy>=0.17', 'packaging', 'tqdm',
                    'portalocker>=2.3.0', 'dill>=0.3.3']
if sys.version_info < (3, 7):
    install_requires.append('dataclasses>=0.6')

setup(
    name='cobaya',
    version=find_version(),
    description='Code for Bayesian Analysis',
    long_description=get_long_description(),
    url="https://cobaya.readthedocs.io",
    project_urls={
        'Source': 'https://github.com/CobayaSampler/cobaya',
        'Tracker': 'https://github.com/CobayaSampler/cobaya/issues',
        'Licensing': 'https://github.com/CobayaSampler/cobaya/blob/master/LICENCE.txt'
    },
    author="Jesus Torrado and Antony Lewis",
    license='LGPL',
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    python_requires='>=3.6.1',
    keywords='montecarlo sampling MCMC cosmology',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=install_requires,
    extras_require={
        'test': ['pytest', 'pytest-forked', 'flaky', 'mpi4py'],
        'gui': ['pyqt5', 'pyside2', 'matplotlib'],
        'docs': extract_docs_requirements()},
    package_data={
        'cobaya': list(chain(*[['%s/*/*.yaml' % folder, '%s/*/*.bibtex' % folder]
                               for folder in subfolders.values()]))},
    entry_points={
        'console_scripts': [
            'cobaya-install=cobaya.install:install_script',
            'cobaya-create-image=cobaya.containers:create_image_script',
            'cobaya-prepare-data=cobaya.containers:prepare_data_script',
            'cobaya-run=cobaya.run:run_script',
            'cobaya-doc=cobaya.doc:doc_script',
            'cobaya-bib=cobaya.bib:bib_script',
            'cobaya-grid-create=cobaya.grid_tools:make_grid_script',
            'cobaya-grid-run=cobaya.grid_tools.runbatch:run',
            'cobaya-run-job=cobaya.grid_tools.runMPI:run_single',
            'cobaya-running-jobs=cobaya.grid_tools.running_jobs:running_jobs',
            'cobaya-delete-jobs=cobaya.grid_tools.delete_jobs:delete_jobs',
            'cobaya-cosmo-generator=cobaya.cosmo_input:gui_script',
        ],
    },
)
