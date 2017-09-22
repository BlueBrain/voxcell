'''setup.py'''
# pylint: skip-file
from __future__ import print_function

import os
import sys

from optparse import Option
import pip
from pip.req import parse_requirements

from distutils import log
from setuptools import setup, find_packages, Command
from setuptools.command import (sdist, build_py, egg_info)
from subprocess import check_call
from voxcellview.version import VERSION as __version__


if sys.version_info < (2, 7):
    sys.exit("Python < 2.7 is no longer supported from version 2.1.0")

here = os.path.dirname(os.path.abspath(__file__))
node_root = os.path.join(here, 'js')
is_repo = os.path.exists(os.path.join(here, '.git'))

npm_path = os.pathsep.join([
    os.path.join(node_root, 'node_modules', '.bin'),
    os.environ.get('PATH', os.defpath),
])

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

log.set_verbosity(log.DEBUG)
log.info('setup.py entered')
log.info('$PATH=%s' % os.environ['PATH'])

LONG_DESCRIPTION = 'viewers for voxcell the brain building framework'


def js_prerelease(command, strict=False):
    """decorator for building minified js/css prior to another command"""
    class DecoratedCommand(command):
        def run(self):
            jsdeps = self.distribution.get_command_obj('jsdeps')
            if not is_repo and all(os.path.exists(t) for t in jsdeps.targets):
                # sdist, nothing to do
                command.run(self)
                return

            try:
                self.distribution.run_command('jsdeps')
            except Exception as e:
                missing = [t for t in jsdeps.targets if not os.path.exists(t)]
                if strict or missing:
                    log.warn('rebuilding js and css failed')
                    if missing:
                        log.error('missing files: %s' % missing)
                    raise e
                else:
                    log.warn('rebuilding js and css failed (not a problem)')
                    log.warn(str(e))
            command.run(self)
            update_package_data(self.distribution)
    return DecoratedCommand


def update_package_data(distribution):
    """update package_data to catch changes during setup"""
    build_py_ = distribution.get_command_obj('build_py')
    # distribution.package_data = find_package_data()
    # re-init build_py options which load package_data
    build_py_.finalize_options()


class NPM(Command):
    '''build NPM package'''
    description = 'install package.json dependencies using npm'
    user_options = []
    node_modules = os.path.join(node_root, 'node_modules')
    targets = [
        os.path.join(here, 'voxcellview', 'static', 'extension.js'),
        os.path.join(here, 'voxcellview', 'static', 'index.js')
    ]

    def initialize_options(self):
        '''initialize_options'''
        pass

    def finalize_options(self):
        '''finalize_options'''
        pass

    def has_npm(self):  # pylint: disable=no-self-use
        '''has_npm'''
        try:
            check_call(['npm', '--version'])
            return True
        except:  # pylint: disable=bare-except
            return False

    def should_run_npm_install(self):
        '''should_run_npm_install'''
        return self.has_npm()

    def run(self):
        '''run'''
        has_npm = self.has_npm()
        if not has_npm:
            log.error("`npm` unavailable.  If you're running this command using "
                      "sudo, make sure `npm` is available to sudo")

        env = os.environ.copy()
        env['PATH'] = npm_path

        if self.should_run_npm_install():
            log.info("Installing build dependencies with npm.  This may take a while...")
            check_call(['npm', 'install'], cwd=node_root, stdout=sys.stdout, stderr=sys.stderr)
            os.utime(self.node_modules, None)

        for t in self.targets:
            if not os.path.exists(t):
                msg = 'Missing file: %s' % t
                if not has_npm:
                    msg += '\nnpm is required to build a development version of widgetsnbextension'
                raise ValueError(msg)

        # update package data in case this created new files
        update_package_data(self.distribution)


setup_args = {
    'name': 'voxcellview',
    'version': __version__,
    'description': 'viewers for voxcell the brain building framework',
    'long_description': LONG_DESCRIPTION,
    'include_package_data': True,
    'data_files': [
        ('share/jupyter/nbextensions/voxcellview', [
            'voxcellview/static/extension.js',
            'voxcellview/static/index.js'
        ]),
    ],
    'install_requires': REQS,
    'packages': find_packages(),
    'zip_safe': False,
    'cmdclass': {
        'build_py': js_prerelease(build_py.build_py),
        'egg_info': js_prerelease(egg_info.egg_info),
        'sdist': js_prerelease(sdist.sdist, strict=True),
        'jsdeps': NPM,
    },

    'author': 'courcol',
    'author_email': 'jean-denis.courcol@epfl.ch',
    'url': 'https://bbp.epfl.ch',
    'keywords': [
        'ipython',
        'jupyter',
        'widgets',
    ],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Framework :: IPython',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
}

setup(**setup_args)
