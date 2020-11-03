#!/usr/bin/env python

""" Distribution configuration """

import imp
import sys

from setuptools import setup, find_packages


if sys.version_info < (2, 7):
    sys.exit("Python < 2.7 is no longer supported from version 1.4.0")

VERSION = imp.load_source("voxcell.version", "voxcell/version.py").VERSION

SONATA_REQUIRES = [
    'libsonata>=0.0.1,<1.0',
]


setup(
    name='voxcell',
    author='NSE Team',
    author_email='bbp-ou-nse@groupes.epfl.ch',
    version=VERSION,
    description='Voxcell is a small library to handle probability'
                ' distributions that have a spatial component and to use them'
                ' to build collection of cells in space.',
    url='https://bbpteam.epfl.ch/project/issues/projects/BRBLD/issues/',
    download_url='https://bbpteam.epfl.ch/repository/devpi/+search?query=name%3Avoxcell',
    license='BBP-internal-confidential',
    install_requires=[
        'future>=0.16',
        'h5py>=2.3,<3.0.0',
        'numpy>=1.9',
        'pandas>=0.24.2',
        'pynrrd>=0.4.0',
        'requests>=2.18',
        'scipy>=0.13',
        'six>=1.0',
    ],
    extras_require={
        'all': SONATA_REQUIRES,
        'sonata': SONATA_REQUIRES,
    },
    packages=find_packages()
)
