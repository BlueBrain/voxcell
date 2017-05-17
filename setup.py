#!/usr/bin/env python
# pylint: skip-file
""" Distribution configuration """

import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import imp
import pip
from pip.req import parse_requirements
from optparse import Option

if sys.version_info < (2, 7):
    sys.exit("Python < 2.7 is no longer supported from version 1.4.0")

options = Option('--workaround')
options.skip_requirements_regex = None
REQ_FILE = './requirements.txt'
# Hack for old pip versions: Versions greater than 1.x
# have a required parameter "sessions" in parse_requierements
if pip.__version__.startswith('1.'):
    install_reqs = parse_requirements(REQ_FILE, options=options)
else:
    from pip.download import PipSession  # pylint:disable=E0611
    options.isolated_mode = False
    install_reqs = parse_requirements(REQ_FILE,  # pylint:disable=E1123
                                      options=options,
                                      session=PipSession)

REQS = [str(ir.req) for ir in install_reqs]

VERSION = imp.load_source("voxcell.version", "voxcell/version.py").VERSION

setup(
    name='voxcell',
    version=VERSION,
    description='code to build collections of cells',
    install_requires=REQS,
    packages=['voxcell'],
    include_package_data=True,
)
