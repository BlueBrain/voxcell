#!/usr/bin/env python
# pylint: disable=R0801
""" Distribution configuration """
# pylint: disable=R0801,F0401,E0611

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


from pip.req import parse_requirements
from optparse import Option

from voxcellview import __version__

OPTIONS = Option("--workaround")
OPTIONS.skip_requirements_regex = None
INSTALL_REQS = parse_requirements("./requirements.txt", options=OPTIONS)
REQS = [str(ir.req) for ir in INSTALL_REQS]

setup(
    name='voxcellview',
    version=__version__,
    description='code to visualize data structures built with voxcell',
    install_requires=REQS,
    packages=['voxcellview'],
    include_package_data=True,
    scripts=[],
)
