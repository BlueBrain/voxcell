#!/usr/bin/env python
# pylint: disable=R0801
""" Distribution configuration """
# pylint: disable=R0801,F0401,E0611
import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


from voxcellview import __version__
BASEDIR = os.path.dirname(os.path.abspath(__file__))
REQS = []
EXTRA_REQS_PREFIX = 'requirements_'
EXTRA_REQS = {}

import pip
from pip.req import parse_requirements
from optparse import Option


def parse_reqs(reqs_file):
    ''' parse the requirements '''
    options = Option('--workaround')
    options.skip_requirements_regex = None
    # Hack for old pip versions: Versions greater than 1.x
    # have a required parameter "sessions" in parse_requierements
    if pip.__version__.startswith('1.'):
        install_reqs = parse_requirements(reqs_file, options=options)
    else:
        from pip.download import PipSession  # pylint:disable=E0611
        options.isolated_mode = False
        install_reqs = parse_requirements(reqs_file,  # pylint:disable=E1123
                                          options=options,
                                          session=PipSession)
    return [str(ir.req) for ir in install_reqs]

REQS = parse_reqs(os.path.join(BASEDIR, 'requirements.txt'))

#look for extra requirements (ex: requirements_bbp.txt)
for file_name in os.listdir(BASEDIR):
    if not file_name.startswith(EXTRA_REQS_PREFIX):
        continue
    base_name = os.path.basename(file_name)
    (extra, _) = os.path.splitext(base_name)
    extra = extra[len(EXTRA_REQS_PREFIX):]
    EXTRA_REQS[extra] = parse_reqs(file_name)


setup(
    name='voxcellview',
    version=__version__,
    description='code to visualize data structures built with voxcell',
    install_requires=REQS,
    extras_require=EXTRA_REQS,
    packages=['voxcellview'],
    include_package_data=True,
    entry_points={'console_scripts':
                  ['install_extension=voxcellview.install:main']}
)
