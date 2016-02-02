#!/usr/bin/env python
# pylint: disable=R0801
""" Distribution configuration """
# pylint: disable=R0801,F0401,E0611

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import pip
from pip.req import parse_requirements
from optparse import Option

from voxcellview import __version__
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

setup(
    name='voxcellview',
    version=__version__,
    description='code to visualize data structures built with voxcell',
    install_requires=REQS,
    packages=['voxcellview'],
    include_package_data=True,
    entry_points={'console_scripts':
                  ['install_extension=voxcellview.install:main']}
)
