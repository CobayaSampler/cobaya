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
    description='Bayesian Analysis in Cosmology',
    long_description=long_description,
    url='https://github.com/JesusTorrado/cobaya',
    author='Jesus Torrado and Antony Lewis',
    license='LGPL (code, except where stated otherwise) and GFDL (docs)',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
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
    install_requires=['numpy', 'scipy >= 0.18', 'pandas', 'pyyaml'],
    entry_points={
        'console_scripts': [
            'cobaya-run=cobaya.run:run_script',
            'cobaya-citation=cobaya.citation:citation_script',
        ],
    },
)
