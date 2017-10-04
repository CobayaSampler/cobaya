# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

from cobaya import __author__, __version__, __url__

setup(
    name='cobaya',
    version=__version__,
    description='Bayesian Analysis in Cosmology',
    long_description=long_description,
    url=__url__,
    author=__author__,
    license='LGPL',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='montecarlo sampling cosmology',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=['numpy', 'scipy >= 0.18', 'pandas', 'pyyaml', 'wget'],
    python_requires='>=2.7, <3',
    entry_points={
        'console_scripts': [
            'cobaya-install=cobaya.install:install_script',
            'cobaya-run=cobaya.run:run_script',
            'cobaya-citation=cobaya.citation:citation_script',
        ],
    },
)
