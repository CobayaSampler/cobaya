# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
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
    description='Bayesian Analysis in Cosmology',
    long_description=get_long_description(),
    url=__url__,
    author=__author__,
    license='LGPL',
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: '
            'GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,!=3.5.*',
    keywords='montecarlo sampling cosmology',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy>=1.12.0', 'scipy >= 1.0', 'pandas>=0.20',
                      'PyYAML>=5.1', 'wget>=3.2', 'imageio>=2.2.0',
                      'GetDist>=0.3.0'],
    extras_require={
        'test': ['pytest', 'pytest-xdist', 'flaky', 'mpi4py']},
    package_data={
        'cobaya': list(chain(*[['%s/*/*.yaml' % folder, '%s/*/*.bibtex' % folder]
                               for folder in subfolders.values()]))},
    entry_points={
        'console_scripts': [
            'cobaya-install=cobaya.install:install_script',
            'cobaya-create-image=cobaya.containers:create_image_script',
            'cobaya-prepare-data=cobaya.containers:prepare_data_script',
            'cobaya-run=cobaya.run:run_script',
            'cobaya-citation=cobaya.citation:citation_script',
            'cobaya-grid-create=cobaya.grid_tools:MakeGridScript',
            'cobaya-grid-run=cobaya.grid_tools.runbatch:run',
            'cobaya-cosmo-generator=cobaya.cosmo_input:gui_script',
        ],
    },
)
