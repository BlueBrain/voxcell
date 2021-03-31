#!/usr/bin/env python

""" Distribution configuration """

import imp
import sys

from setuptools import setup, find_packages


VERSION = imp.load_source("voxcell.version", "voxcell/version.py").VERSION

# keep this to avoid breaking the pip API should be removed in 2.9.0
SONATA_REQUIRES = [
]


setup(
    name='voxcell',
    author='NSE Team',
    author_email='bbp-ou-nse@groupes.epfl.ch',
    version=VERSION,
    description='Voxcell is a small library to handle probability'
                ' distributions that have a spatial component and to use them'
                ' to build collection of cells in space.',
    url='https://bbpteam.epfl.ch/documentation/projects/voxcell',
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/NSETM/issues",
        "Source": "ssh://bbpcode.epfl.ch/nse/voxcell",
    },
    license='BBP-internal-confidential',
    install_requires=[
        'future>=0.16',
        'h5py>=3.1.0',
        'numpy>=1.9',
        'pandas>=0.24.2',
        'pynrrd>=0.4.0',
        'requests>=2.18',
        'scipy>=0.13',
    ],
    extras_require={
        'all': SONATA_REQUIRES,
        'sonata': SONATA_REQUIRES,
    },
    packages=find_packages(),
    python_requires='>=3.6',
)
