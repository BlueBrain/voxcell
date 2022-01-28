#!/usr/bin/env python

"""Distribution configuration."""

import importlib.util

from setuptools import setup, find_packages

spec = importlib.util.spec_from_file_location(
    "voxcell.version",
    "voxcell/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.__version__

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
        "Source": "https://bbpgitlab.epfl.ch/nse/voxcell.git",
    },
    license='BBP-internal-confidential',
    install_requires=[
        'h5py>=3.1.0',
        'numpy>=1.9',
        'pandas>=0.24.2',
        'pynrrd>=0.4.0',
        'requests>=2.18',
        'scipy>=1.2.0',
    ],
    extras_require={
        'all': SONATA_REQUIRES,
        'sonata': SONATA_REQUIRES,
    },
    packages=find_packages(),
    python_requires='>=3.7',
)
