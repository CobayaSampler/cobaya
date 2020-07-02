# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from os import path
from itertools import chain
# Package data and conventions
from cobaya import __author__, __version__, __url__
from cobaya.conventions import subfolders


# Get the long description from the README file
def get_long_description():
    with open(path.join(path.abspath(path.dirname(__file__)), 'README.rst'),
              encoding='utf-8') as f:
        lines = f.readlines()
        i = -1
        while '=====' not in lines[i]:
            i -= 1
        return "".join(lines[:i])


setup(
    name='cobaya',
    version=__version__,
    description='Code for Bayesian Analysis',
    long_description=get_long_description(),
    url=__url__,
    project_urls={
        'Source': 'https://github.com/CobayaSampler/cobaya',
        'Tracker': 'https://github.com/CobayaSampler/cobaya/issues',
        'Licensing': 'https://github.com/CobayaSampler/cobaya/blob/master/LICENCE.txt'
    },
    author=__author__,
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
        'Programming Language :: Python :: 3.8'
    ],
    python_requires='>=3.6.1',
    keywords='montecarlo sampling MCMC cosmology',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy>=1.12.0', 'scipy>=1.0', 'pandas>=1.0.1',
                      'PyYAML>=5.1', 'requests>=2.18', 'py-bobyqa>=1.1',
                      'GetDist>=1.1.1', 'fuzzywuzzy>=0.17', 'packaging'],
    extras_require={
        'test': ['pytest', 'pytest-forked', 'flaky', 'mpi4py'],
        'gui': ['pyqt5', 'pyside2']},
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
            'cobaya-cosmo-generator=cobaya.cosmo_input:gui_script',
        ],
    },
)
