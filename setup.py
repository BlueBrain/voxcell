#!/usr/bin/env python

""" Distribution configuration """

import imp
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.version_info < (2, 7):
    sys.exit("Python < 2.7 is no longer supported from version 1.4.0")

VERSION = imp.load_source("voxcell.version", "voxcell/version.py").VERSION

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
        'h5py>=2.3',
        'numpy>=1.9',
        'pandas>=0.17',
        'pynrrd>=0.2',
        'scipy>=0.13',
        'six>=1.0',
    ],
    packages=[
        'voxcell',
        'voxcell.utils',
    ],
    test_suite='nose.collector'
)
