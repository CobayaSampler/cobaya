# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cobaya',
    version='0.1',
    description='A Monte Carlo sampler ready for Cosmology',
    long_description=long_description,
    url='https://github.com/JesusTorrado/cobaya',
    author='Jesus Torrado and Antony Lewis',
    license='LGPL [PROVISIONAL!]',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: LGPL [PROVISIONAL!]',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
#        'Programming Language :: Python :: 3',
    ],
    keywords='cosmology montecarlo sampling',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy', 'scipy >= 0.18', 'matplotlib', 'pandas', 'pyyaml',
                      #'getdist' # not for now: using modified version
                      #mpi4py
                     ],
    entry_points={
        'console_scripts': [
            'cobaya-run=cobaya.run:run_script',
            'cobaya-citation=cobaya.citation:citation_script',
        ],
    },
)
